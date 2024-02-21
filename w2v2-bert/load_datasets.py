'''
The following code was heavily copied from this repository: 
    https://github.com/fengredrum/finetune-whisper-lora
'''

import gc

from datasets import (load_dataset, concatenate_datasets,
                      IterableDatasetDict, DatasetDict, Audio)


def load_common_voice(
        full_name="mozilla-foundation/common_voice_16_0",
        language_abbr="mn",
        sampling_rate=16000,
        streaming=True,
        cache_dir="~/.cache/huggingface/datasets",
        use_valid_to_train=True,
        test_only=False,
):

    # Load test data
    ds_test = load_dataset(
        full_name, language_abbr, split="test",
        cache_dir=cache_dir, token=True, trust_remote_code=True)

    # Load train data
    if not test_only:
        ds_train = load_dataset(
            full_name, language_abbr, split="train",
            cache_dir=cache_dir, token=True, trust_remote_code=True)

    # Use validation data to train
    if use_valid_to_train and not test_only:
        ds_valid = load_dataset(
            full_name, language_abbr, split="validation",
            cache_dir=cache_dir, token=True, trust_remote_code=True)
        ds_train = concatenate_datasets([ds_train, ds_valid])
        del ds_valid

    if streaming:
        ds = IterableDatasetDict()
        ds["test"] = ds_test.to_iterable_dataset()
        if not test_only:
            ds["train"] = ds_train.to_iterable_dataset()
            del ds_train
    else:
        ds = DatasetDict()
        ds["test"] = ds_test
        if not test_only:
            ds["train"] = ds_train
            del ds_train

    del ds_test

    # Remove unnecessary columns
    ds = ds.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", "variant"])
    # Convert sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    gc.collect()
    return ds


def load_process_datasets(datasets_settings,
                          streaming=True,
                          cache_dir="~/.cache/huggingface/datasets",
                          num_train_samples=1000,
                          num_test_samples=1000,
                          test_only=False,
                          sampling_rate=16000,
                          buffer_size=500,
                          seed=42,
                          ):

    if streaming:
        ds = IterableDatasetDict()
    else:
        ds = DatasetDict()

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(f"Processing dataset: {name} with {kwargs}...")
        ds_tmp = None

        if name == "common_voice":
            ds_tmp = load_common_voice(sampling_rate=sampling_rate,
                                       streaming=streaming, cache_dir=cache_dir, test_only=test_only, **kwargs)

        # Collect datasets
        if ds_tmp is not None:
            print(f"{name} sample: ", next(iter(ds_tmp["test"])))
            test_list.append(ds_tmp["test"])
            if not test_only:
                train_list.append(ds_tmp["train"])

    # Concatenate datasets
    ds["test"] = concatenate_datasets(test_list)
    if not test_only:
        ds["train"] = concatenate_datasets(train_list)

    if streaming:
        ds = ds.shuffle(seed, buffer_size=buffer_size)

        ds["test"] = ds["test"].take(num_test_samples)
        if not test_only:
            ds["train"] = ds["train"].take(num_train_samples)
    else:
        ds = ds.shuffle(seed)

        num_test_samples = min(num_test_samples, ds["test"].num_rows)
        ds["test"] = ds["test"].select(range(num_test_samples))
        if not test_only:
            num_train_samples = min(num_train_samples, ds["train"].num_rows)
            ds["train"] = ds["train"].select(range(num_train_samples))

    return ds


if __name__ == "__main__":
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
