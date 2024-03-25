import gc
import torch
import argparse
import numpy as np
import evaluate

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from load_datasets import load_process_datasets
from utils import prepare_dataset, DataCollatorCTCWithPadding

datasets_settings = [
    [
        "mdcc",
        {
            "use_valid_to_train": False,
        },
        {
            "target_lang": "yue",
            "chars_to_remove": r"[\,\?\.\!\-\;\:\%\‘\’\“\”\»\«\…\'\"\_\،\؛\؟\ـ\{\}]",
        },
    ],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument(
        "--model_name_or_path", default="logs/mms-mdcc/checkpoint-1500/"
    )
    parser.add_argument("--metric", default="cer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", default=16, type=int)
    # Dataset setups
    parser.add_argument("--num_test_samples", default=2000, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=4, type=int)
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

    # Load pretrained model
    if args.device == "cuda" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
        args.device = "cpu"

    target_lang = datasets_settings[0][-1]["target_lang"]
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path, ignore_mismatched_sizes=True, torch_dtype=dtype
    ).to(args.device)
    model.eval()
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=target_lang,
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

    if args.streaming:
        map_kwargs = {}
        training_kwargs = {}
    else:
        map_kwargs = {"num_proc": args.num_proc}
        training_kwargs = {"group_by_length": False}

    ds = ds.map(
        prepare_dataset,
        remove_columns=ds["test"].column_names,
        fn_kwargs={"processor": processor},
        **map_kwargs,
    ).with_format("torch")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    eval_dataloader = DataLoader(
        ds["test"],
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    # Load metric
    metric = evaluate.load(args.metric)

    for step, batch in enumerate(tqdm(eval_dataloader)):

        if args.device == "cuda" and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = model(batch["input_values"].to("cuda")).logits
        else:
            with torch.no_grad():
                logits = model(batch["input_values"].to("cpu")).logits

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
