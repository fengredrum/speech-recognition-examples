import gc
import torch
import random
import numpy as np
import evaluate
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric, Audio
from transformers import (Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer,
                          SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor)

from load_datasets import load_process_datasets

datasets_settings = [
    ["common_voice",
     {"full_name": "mozilla-foundation/common_voice_16_0",
      "language_abbr": "mn", "use_valid_to_train": False, }
     ],
    ["common_voice",
     {"full_name": "mozilla-foundation/common_voice_16_0",
      "language_abbr": "ml", "use_valid_to_train": False, }
     ],
]

if __name__ == "__main__":

    seed = 27
    num_proc=4
    batch_size = 32
    num_test_samples = 1000
    device = "cuda"
    model_name_or_path = "w2v-bert-CV16.0/checkpoint-1000"

    # Load dataset
    ds = load_process_datasets(
        datasets_settings,
        num_test_samples=num_test_samples,
        test_only=True,
        streaming=False,
        seed=seed,
    )

    # Load processor
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        model_name_or_path)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Convert sampling rate
    ds = ds.cast_column(
        "audio", Audio(sampling_rate=16_000))

    rand_int = random.randint(0, num_test_samples-1)
    print("Target text:",
          ds["test"][rand_int]["sentence"])
    print("Input array shape:",
          ds["test"][rand_int]["audio"]["array"].shape)
    print("Sampling rate:",
          ds["test"][rand_int]["audio"]["sampling_rate"])

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    ds = ds.map(
        prepare_dataset, remove_columns=ds["test"].column_names, num_proc=num_proc)

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
    
    eval_dataloader = DataLoader(
        ds["test"], batch_size=batch_size, collate_fn=data_collator)

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
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    metric = evaluate.load("wer")

    for step, batch in enumerate(tqdm(eval_dataloader)):

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits = model(batch["input_features"].to(device)).logits

        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        decoded_preds = processor.batch_decode(pred_ids)
        decoded_labels = processor.batch_decode(labels)
        metric.add_batch(predictions=decoded_preds,
                                    references=decoded_labels)
        del logits, labels, batch, pred_ids
        gc.collect()

    wer = 100 * metric.compute()
    print(f"{wer=}")
    print(f"pred: {decoded_preds}")
    print(f"label: {decoded_labels}")
