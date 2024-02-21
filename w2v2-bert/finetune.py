import torch
import json
import random
import argparse
import numpy as np

from datasets import load_dataset, load_metric, Audio
from transformers import TrainingArguments, Trainer
from transformers import (Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer,
                          SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from utils import remove_special_characters, extract_all_chars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--num_train_samples", default=2000, type=int)
    parser.add_argument("--num_test_samples", default=500, type=int)
    # parser.add_argument("--max_input_length", default=30.0, type=float)
    # parser.add_argument("--streaming", default=False, type=bool)
    parser.add_argument("--num_proc", default=4, type=int)
    # Finetuning setups
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--fp16", default=True, type=bool)

    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)

    args = parser.parse_args()
    print(f"Settings: {args}")

    model_name_or_path = "facebook/w2v-bert-2.0"

    # Load dataset
    dataset_name = "mozilla-foundation/common_voice_16_0"
    language_abbr = "mn"
    common_voice_train = load_dataset(
        dataset_name, language_abbr, split="train", use_auth_token=True)
    common_voice_test = load_dataset(
        dataset_name, language_abbr, split="test", use_auth_token=True)
    common_voice_train = common_voice_train.select(
        range(args.num_train_samples))
    common_voice_test = common_voice_test.select(range(args.num_test_samples))

    # Remove unnecessary columns
    common_voice_train = common_voice_train.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", "variant"])
    common_voice_test = common_voice_test.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", "variant"])

    # Remove special characters and normalize text
    chars_to_remove = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
    common_voice_train = common_voice_train.map(remove_special_characters, fn_kwargs={
                                                "chars_to_remove_regex": chars_to_remove}, num_proc=args.num_proc)
    common_voice_test = common_voice_test.map(remove_special_characters, fn_kwargs={
        "chars_to_remove_regex": chars_to_remove}, num_proc=args.num_proc)

    # Build vocabulary and load into tokenizer
    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1,
                                         keep_in_memory=True, remove_columns=common_voice_train.column_names, num_proc=args.num_proc)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1,
                                       keep_in_memory=True, remove_columns=common_voice_test.column_names, num_proc=args.num_proc)

    vocab_list = list(set(vocab_train["vocab"][0])
                      | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    # Preparing for finetuning
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        model_name_or_path)

    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Convert sampling rate
    common_voice_train = common_voice_train.cast_column(
        "audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column(
        "audio", Audio(sampling_rate=16_000))

    rand_int = random.randint(0, args.num_train_samples-1)
    print("Target text:",
          common_voice_train[rand_int]["sentence"])
    print("Input array shape:",
          common_voice_train[rand_int]["audio"]["array"].shape)
    print("Sampling rate:",
          common_voice_train[rand_int]["audio"]["sampling_rate"])

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    common_voice_train = common_voice_train.map(
        prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=args.num_proc)
    common_voice_test = common_voice_test.map(
        prepare_dataset, remove_columns=common_voice_test.column_names, num_proc=args.num_proc)

    # Create Collator

    @dataclass
    class DataCollatorCTCWithPadding:

        processor: Wav2Vec2BertProcessor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_features": feature["input_features"]}
                              for feature in features]
            label_features = [{"input_ids": feature["labels"]}
                              for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )

            labels_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding,
                return_tensors="pt",
            )
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)

    metric = load_metric(args.metric)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -
                       100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        metric_result = metric.compute(
            predictions=pred_str, references=label_str)

        return {args.metric: metric_result}

    # Load pretrained model
    model = Wav2Vec2BertForCTC.from_pretrained(
        model_name_or_path,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).to(args.device)

    repo_name = "w2v-bert-CV16.0"
    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        learning_rate=5e-5,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model=args.metric,
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
    )

    trainer.train()
