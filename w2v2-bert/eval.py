import gc
import torch
import random
import numpy as np
import evaluate

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer,
                          SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor)

from load_datasets import load_process_datasets
from utils import prepare_dataset, DataCollatorCTCWithPadding

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
    num_test_samples = 100
    device = "cuda"
    model_name_or_path = "facebook/w2v-bert-2.0"

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

    rand_int = random.randint(0, num_test_samples-1)
    print("Target text:",
          ds["test"][rand_int]["sentence"])
    print("Input array shape:",
          ds["test"][rand_int]["audio"]["array"].shape)
    print("Sampling rate:",
          ds["test"][rand_int]["audio"]["sampling_rate"])

    ds = ds.map(
        prepare_dataset, remove_columns=ds["test"].column_names, 
        fn_kwargs={"processor": processor},
        num_proc=num_proc, 
        )

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)
    
    eval_dataloader = DataLoader(
        ds["test"], batch_size=batch_size, collate_fn=data_collator)

    # Load pretrained model
    # model = Wav2Vec2BertForCTC.from_pretrained(
    #     model_name_or_path,
    #     attention_dropout=0.0,
    #     hidden_dropout=0.0,
    #     feat_proj_dropout=0.0,
    #     mask_time_prob=0.0,
    #     layerdrop=0.0,
    #     ctc_loss_reduction="mean",
    #     add_adapter=True,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     vocab_size=len(processor.tokenizer),
    #     torch_dtype=torch.float16,
    # ).to(device)
    # model.eval()

    metric = evaluate.load("wer")

    for step, batch in enumerate(tqdm(eval_dataloader)):
        pass

    #     with torch.cuda.amp.autocast():
    #         with torch.no_grad():
    #             logits = model(batch["input_features"].to(device)).logits

    #     labels = batch["labels"].cpu().numpy()
    #     labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    #     pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

    #     decoded_preds = processor.batch_decode(pred_ids)
    #     decoded_labels = processor.batch_decode(labels)
    #     metric.add_batch(predictions=decoded_preds,
    #                                 references=decoded_labels)
    #     del logits, labels, batch, pred_ids
    #     gc.collect()

    # wer = 100 * metric.compute()
    # print(f"{wer=}")
    # print(f"pred: {decoded_preds}")
    # print(f"label: {decoded_labels}")

    print(batch["input_features"])
    print(batch["labels"])