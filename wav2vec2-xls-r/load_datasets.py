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


def load_common_language(
    cache_dir="~/.cache/huggingface/datasets",
    use_valid_to_train=True,
    test_only=False,
):
    ds = DatasetDict()

    # Load test data
    ds["test"] = load_dataset(
        "common_language",
        split="test",
        cache_dir=cache_dir,
    )

    # Load train data
    if not test_only:
        ds["train"] = load_dataset(
            "common_language",
            split="train",
            cache_dir=cache_dir,
        )

    # Use validation data to train
    if use_valid_to_train and not test_only:
        ds["valid"] = load_dataset(
            "common_language",
            split="validation",
            cache_dir=cache_dir,
        )
        ds["train"] = concatenate_datasets([ds["train"], ds["valid"]])
        del ds["valid"]

    # Remove unnecessary columns
    ds = ds.remove_columns(
        [
            "client_id",
            "age",
            "gender",
        ]
    )

    gc.collect()
    return ds


def load_process_datasets(
    datasets_settings,
    streaming=True,
    cache_dir="~/.cache/huggingface/datasets",
    num_train_samples=1000,
    num_test_samples=1000,
    test_only=False,
    sampling_rate=16000,
    buffer_size=500,
    num_proc=4,
    seed=27,
):

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(f"Processing dataset: {name} with {kwargs}...")
        ds_tmp = None

        if name == "common_language":
            ds_tmp = load_common_language(
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
