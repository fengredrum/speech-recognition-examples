import argparse
import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-base")
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--max_duration_in_seconds", default=1.0, type=float)
    # Finetuning setups
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--fp16", default=True, type=bool)

    args = parser.parse_args()
    print(f"Settings: {args}")

    # Load dataset
    dataset = load_dataset("superb", "ks")
    print(dataset)

    labels = dataset["train"].features["label"].names
    print(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label


    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)


    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * args.max_duration_in_seconds),
            truncation=True,
        )
        return inputs


    encoded_dataset = dataset.map(
        preprocess_function, remove_columns=["audio", "file"], batched=True
    )


    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to("cuda")


    metric = load_metric("accuracy")


    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    experiment_name = model_name.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-ks",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
