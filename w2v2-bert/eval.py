import gc
import torch
import random
import argparse
import numpy as np
import evaluate

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
)

from load_datasets import load_process_datasets
from utils import prepare_dataset, DataCollatorCTCWithPadding

datasets_settings = [
    [
        "common_voice",
        {
            "full_name": "mozilla-foundation/common_voice_16_0",
            "language_abbr": "mn",
            "use_valid_to_train": False,
        },
    ],
    [
        "common_voice",
        {
            "full_name": "mozilla-foundation/common_voice_16_0",
            "language_abbr": "ml",
            "use_valid_to_train": False,
        },
    ],
]

model_name_or_path = "facebook/w2v-bert-2.0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", default=32, type=int)
    # Dataset setups
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=2, type=int)
    parser.add_argument("--seed", default=27, type=int)

    args = parser.parse_args()
    print(f"Settings: {args}")

    # Load dataset
    ds = load_process_datasets(
        datasets_settings,
        num_test_samples=args.num_test_samples,
        test_only=True,
        streaming=args.streaming,
        seed=args.seed,
    )
    print("Dataset info: ", ds)

    # Load processor
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name_or_path)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    if args.streaming:
        ds = ds.map(
            prepare_dataset,
            remove_columns=list(next(iter(ds.values())).features),
            fn_kwargs={"processor": processor},
        ).with_format("torch")
    else:
        ds = ds.map(
            prepare_dataset,
            remove_columns=ds["test"].column_names,
            fn_kwargs={"processor": processor},
            num_proc=args.num_proc,
        ).with_format("torch")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    eval_dataloader = DataLoader(
        ds["test"],
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    # Load metric
    metric = evaluate.load(args.metric)

    # Load pretrained model
    if args.device == "cuda" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
        args.device = "cpu"

# TODO remove unnecessary params
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
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    for step, batch in enumerate(tqdm(eval_dataloader)):

        if args.device == "cuda" and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = model(batch["input_features"].to("cuda")).logits
        else:
            with torch.no_grad():
                logits = model(batch["input_features"].to("cpu")).logits

        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        decoded_preds = processor.batch_decode(pred_ids)
        decoded_labels = processor.batch_decode(labels)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        del logits, labels, batch, pred_ids
        gc.collect()

    wer = 100 * metric.compute()
    print(f"{wer=}")
    print(f"pred: {decoded_preds}")
    print(f"label: {decoded_labels}")
