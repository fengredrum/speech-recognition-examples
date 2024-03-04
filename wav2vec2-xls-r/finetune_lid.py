import argparse
import numpy as np
import evaluate

from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

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
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--num_train_samples", default=None)
    parser.add_argument("--num_test_samples", default=None)
    parser.add_argument("--max_duration_in_seconds", default=16.0, type=float)
    parser.add_argument("--min_duration_in_seconds", default=2.0, type=float)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=2, type=int)
    parser.add_argument("--seed", default=27, type=int)
    # Finetuning setups
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--fp16", default=True, type=bool)

    parser.add_argument("--warmup_ratio", default=0.1, type=int)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
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

    ds = ds.rename_column('language', 'label')

    labels = ds["train"].features["label"].names
    print(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)


    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * args.max_duration_in_seconds),
            truncation=True,
        )
        return inputs


    encoded_dataset = ds.map(
        preprocess_function, remove_columns=["audio", "path", "sentence"], batched=True, num_proc=args.num_proc
    )


    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to(args.device)


    metric = evaluate.load(args.metric)


    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    experiment_name = args.model_name_or_path.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"logs/{experiment_name}-finetuned-ks",
        learning_rate=1e-4,
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
        metric_for_best_model=args.metric,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()