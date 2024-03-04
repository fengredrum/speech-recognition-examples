import argparse
import numpy as np
import evaluate

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    AutoConfig,
    TrainingArguments,
    Trainer,
)

import torch
import evaluate

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


import transformers
from transformers import Wav2Vec2Processor

from datasets import IterableDatasetDict

from load_datasets import load_process_datasets

# TODO Move to ArgumentParser
datasets_settings = [
    [
        "common_language",
        {
            "use_valid_to_train": False,
        },
    ],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--pooling_mode", default="mean")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--num_train_samples", default=None)
    parser.add_argument("--num_test_samples", default=None)
    parser.add_argument("--max_duration_in_seconds", default=16.0, type=float)
    parser.add_argument("--min_duration_in_seconds", default=2.0, type=float)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=4, type=int)
    parser.add_argument("--seed", default=27, type=int)
    # Finetuning setups
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--fp16", default=True, type=bool)

    parser.add_argument("--warmup_ratio", default=0.1, type=int)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)

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

    # Preparing for finetuning
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model_name_or_path
    )
    target_sampling_rate = feature_extractor.sampling_rate
    print(f"Target sampling rate: {target_sampling_rate}")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    output_column = "language"

    label_list = ds["train"].features[output_column].names
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, "pooling_mode", args.pooling_mode)

    def prepare_dataset(batch, processor):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_features"]) / audio["sampling_rate"]

        batch["labels"] = batch[output_column]
        return batch

    if args.streaming:
        ds_tmp = ds
        ds = IterableDatasetDict()
        ds["test"] = ds_tmp["test"].to_iterable_dataset()
        ds["train"] = ds_tmp["train"].to_iterable_dataset()
        del ds_tmp

    ds = ds.map(
        prepare_dataset,
        remove_columns=ds["train"].column_names,
        fn_kwargs={"processor": processor},
        num_proc=args.num_proc,
        # batch_size=1000,
        # batched=True,
    )

    print(ds)

    @dataclass
    class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

    class Wav2Vec2ClassificationHead(nn.Module):
        """Head for wav2vec classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config

            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Wav2Vec2ClassificationHead(config)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        def merged_strategy(self, hidden_states, mode="mean"):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
                )

            return outputs

        def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
        ):
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @dataclass
    class DataCollatorAudioCLFWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            label_features = [feature["labels"] for feature in features]

            d_type = torch.long if isinstance(label_features[0], int) else torch.float

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch["labels"] = torch.tensor(label_features, dtype=d_type)

            return batch

    data_collator = DataCollatorAudioCLFWithPadding(processor=processor, padding=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    ).to("cpu")
    model.freeze_feature_extractor()

    experiment_name = args.model_name_or_path.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"logs/{experiment_name}-finetuned-lid",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        max_steps=1000,
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=2,
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
