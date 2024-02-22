"""
The following code was heavily copied from this repository: 
    https://github.com/fengredrum/finetune-whisper-lora
"""

import gc
import json

from datasets import (
    load_dataset,
    concatenate_datasets,
    IterableDatasetDict,
    DatasetDict,
    Audio,
)

from utils import remove_special_characters, extract_all_chars

chars_to_remove = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�'\»\«]"


def load_common_voice(
    full_name="mozilla-foundation/common_voice_16_0",
    language_abbr="mn",
    cache_dir="~/.cache/huggingface/datasets",
    use_valid_to_train=True,
    test_only=False,
):

    ds = DatasetDict()

    # Load test data
    ds["test"] = load_dataset(
        full_name,
        language_abbr,
        split="test",
        cache_dir=cache_dir,
        token=True,
        trust_remote_code=True,
    )

    # Load train data
    if not test_only:
        ds["train"] = load_dataset(
            full_name,
            language_abbr,
            split="train",
            cache_dir=cache_dir,
            token=True,
            trust_remote_code=True,
        )

    # Use validation data to train
    if use_valid_to_train and not test_only:
        ds["valid"] = load_dataset(
            full_name,
            language_abbr,
            split="validation",
            cache_dir=cache_dir,
            token=True,
            trust_remote_code=True,
        )
        ds["train"] = concatenate_datasets([ds["train"], ds["valid"]])
        del ds["valid"]

    # Remove unnecessary columns
    ds = ds.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
            "variant",
        ]
    )

    gc.collect()
    return ds


def load_process_datasets(
    datasets_settings,
    streaming=True,
    cache_dir="~/.cache/huggingface/datasets",
    vocab_dir="vocab.json",
    num_train_samples=1000,
    num_test_samples=1000,
    test_only=False,
    sampling_rate=16000,
    buffer_size=500,
    num_proc=4,
    seed=42,
):

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(f"Processing dataset: {name} with {kwargs}...")
        ds_tmp = None

        if name == "common_voice":
            ds_tmp = load_common_voice(
                cache_dir=cache_dir, test_only=test_only, **kwargs
            )

        # Collect datasets
        if ds_tmp is not None:
            print(f"{name} sample: ", next(iter(ds_tmp["test"])))
            test_list.append(ds_tmp["test"])
            if not test_only:
                train_list.append(ds_tmp["train"])

    # Concatenate datasets
    ds = DatasetDict()
    ds["test"] = concatenate_datasets(test_list)
    if not test_only:
        ds["train"] = concatenate_datasets(train_list)

    # Shuffle and select
    ds = ds.shuffle(seed)
    if num_test_samples:
        num_test_samples = min(num_test_samples, ds["test"].num_rows)
        ds["test"] = ds["test"].select(range(num_test_samples))
    if not test_only:
        if num_train_samples:
            num_train_samples = min(num_train_samples, ds["train"].num_rows)
            ds["train"] = ds["train"].select(range(num_train_samples))

        # Remove special characters and normalize text
        ds = ds.map(
            remove_special_characters,
            fn_kwargs={"chars_to_remove_regex": chars_to_remove},
            num_proc=num_proc,
        )

        # Build vocabulary and load into tokenizer
        vocab_text = ds.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=ds["train"].column_names,
            num_proc=num_proc,
        )

        vocab_list = list(
            set(vocab_text["train"]["vocab"][0]) | set(vocab_text["test"]["vocab"][0])
        )
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        print(vocab_dict)

        with open(vocab_dir, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)

    if streaming:
        ds_tmp = ds
        ds = IterableDatasetDict()
        ds["test"] = ds_tmp["test"].to_iterable_dataset()
        if not test_only:
            ds["train"] = ds_tmp["train"].to_iterable_dataset()
        del ds_tmp

    # Convert sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return ds


if __name__ == "__main__":
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

    num_train_samples = 50000
    num_test_samples = 20000

    ds = load_process_datasets(
        datasets_settings,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        test_only=False,
        streaming=False,
        seed=2,
    )
    print(ds)

    print(next(iter(ds["test"])))
