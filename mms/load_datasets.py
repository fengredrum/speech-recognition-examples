"""
The following code was heavily copied from this repository: 
    https://github.com/fengredrum/finetune-whisper-lora
"""

import os
import gc
import json

from datasets import (
    load_dataset,
    concatenate_datasets,
    IterableDatasetDict,
    DatasetDict,
    Dataset,
    Audio,
)

from utils import remove_special_characters, extract_all_chars


def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def create_dataset(
    dataset_dir,
    ds_keys,
    audio_paths,
    transcription_texts,
    cache_dir,
    use_valid_to_train,
    test_only,
):

    ds = DatasetDict()

    for key in ds_keys:
        dataset_dict = {"audio": audio_paths[key], "sentence": transcription_texts[key]}
        ds_tmp = Dataset.from_dict(dataset_dict)

        json_dir = dataset_dir + f"{key}.json"
        if not os.path.exists(json_dir):
            ds_tmp.to_json(json_dir, index=False)

        ds[key] = load_dataset(
            "json",
            data_files=dataset_dir + f"/{key}.json",
            split="train",
            features=ds_tmp.features,
            cache_dir=cache_dir,
        )

    del ds_tmp
    gc.collect()

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    return ds


def load_mdcc(
    dataset_root,
    cache_dir="~/.cache/huggingface/datasets",
    use_valid_to_train=False,
    test_only=False,
):
    dataset_dir = dataset_root + "mdcc/"

    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "valid", "test"]

    audio_paths, transcription_texts = {}, {}
    for key in ds_keys:
        filelist = dataset_dir + f"cnt_asr_{key}_metadata.csv"
        filepaths_and_text = load_filepaths_and_text(filelist)
        filepaths_and_text[0].append("transcription")
        audio_paths[key], transcription_texts[key] = [], []
        for i in range(1, len(filepaths_and_text)):
            audio_path = dataset_dir + filepaths_and_text[i][0][2:]
            audio_paths[key].append(audio_path)

            transcription_path = dataset_dir + filepaths_and_text[i][1][2:]
            with open(transcription_path, encoding="utf-8") as f:
                transcription = [line.strip() for line in f][0]
            # filepaths_and_text[i].append(transcription)
            transcription_texts[key].append(transcription)

    ds = create_dataset(
        dataset_dir=dataset_dir,
        ds_keys=ds_keys,
        audio_paths=audio_paths,
        transcription_texts=transcription_texts,
        cache_dir=cache_dir,
        use_valid_to_train=use_valid_to_train,
        test_only=test_only,
    )

    return ds


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
    dataset_root="../datasets/",
    cache_dir="~/.cache/huggingface/datasets",
    vocab_dir="vocab.json",
    num_train_samples=1000,
    num_test_samples=1000,
    test_only=False,
    sampling_rate=16000,
    num_proc=4,
    seed=42,
):

    train_list, test_list = [], []
    for name, kwargs, settings in datasets_settings:
        print(f"Processing dataset: {name} with {kwargs}...")
        ds_tmp = None

        if name == "common_voice":
            ds_tmp = load_common_voice(
                cache_dir=cache_dir, test_only=test_only, **kwargs
            )
        elif name == "mdcc":
            ds_tmp = load_mdcc(
                dataset_root=dataset_root,
                cache_dir=cache_dir,
                test_only=test_only,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Can not load {name} dataset!")

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
            fn_kwargs={"chars_to_remove_regex": settings["chars_to_remove"]},
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

        new_vocab_dict = {settings["target_lang"]: vocab_dict}
        with open(vocab_dir, "w") as vocab_file:
            json.dump(new_vocab_dict, vocab_file)

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
