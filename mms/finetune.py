import argparse
import numpy as np
import evaluate

from transformers import TrainingArguments, Trainer
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from utils import is_audio_in_length_range, prepare_dataset, DataCollatorCTCWithPadding
from load_datasets import load_process_datasets


# TODO Move to ArgumentParser
datasets_settings = [
    [
        "common_voice",
        {
            "full_name": "mozilla-foundation/common_voice_16_0",
            "language_abbr": "ug",
            "use_valid_to_train": True,
        },
        {
            "target_lang": "uig",
            "chars_to_remove": r"[\,\?\.\!\-\;\:\%\‘\’\“\”\»\«\…\'\"\_\،\؛\؟\ـ]",
        },
    ],
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--model_name_or_path", default="facebook/mms-1b-all")
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--num_train_samples", default=None, type=int)
    parser.add_argument("--num_test_samples", default=None, type=int)
    parser.add_argument("--max_duration_in_seconds", default=30.0, type=float)
    parser.add_argument("--min_duration_in_seconds", default=2.0, type=float)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=4, type=int)
    parser.add_argument("--seed", default=27, type=int)
    # Finetuning setups
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--train_batch_size", default=12, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--fp16", default=True, type=bool)

    parser.add_argument("--warmup_ratio", default=0.1, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)

    args = parser.parse_args()
    print(f"Settings: {args}")

    # Load dataset
    ds = load_process_datasets(
        datasets_settings,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        test_only=False,
        streaming=args.streaming,
        seed=args.seed,
    )
    print("Dataset info: ", ds)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=datasets_settings[0][-1]["target_lang"],
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    ds_sample = next(iter(ds["train"]))
    print("Target text:", ds_sample["sentence"])
    print("Input array shape:", ds_sample["audio"]["array"].shape)
    print("Sampling rate:", ds_sample["audio"]["sampling_rate"])

    if args.streaming:
        map_kwargs = {}
        training_kwargs = {}
    else:
        map_kwargs = {"num_proc": args.num_proc}
        training_kwargs = {"group_by_length": False}

    ds = ds.map(
        prepare_dataset,
        remove_columns=ds["train"].column_names,
        fn_kwargs={"processor": processor},
        **map_kwargs,
    )

    ds = ds.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
        fn_kwargs={
            "min_input_length": args.min_duration_in_seconds,
            "max_input_length": args.max_duration_in_seconds,
        },
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    ).to(args.device)

    model.init_adapter_layers()
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    metric = evaluate.load(args.metric)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        metric_result = metric.compute(predictions=pred_str, references=label_str)

        return {args.metric: metric_result}

    repo_name = "logs/mms-CV16.0"
    training_args = TrainingArguments(
        output_dir=repo_name,
        learning_rate=1e-3,
        warmup_ratio=args.warmup_ratio,
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
        **training_kwargs,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
