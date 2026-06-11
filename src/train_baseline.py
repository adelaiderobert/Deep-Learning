from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_TO_ID = {"non_hateful": 0, "hateful": 1}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def next_run_dir(base_output_dir: Path, run_tag: str | None = None) -> Path:
    if run_tag:
        run_dir = base_output_dir / run_tag
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        return run_dir

    run_idx = 1
    while (base_output_dir / f"run{run_idx}").exists():
        run_idx += 1
    return base_output_dir / f"run{run_idx}"


def load_split(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    return Dataset.from_pandas(df[["id", "text", "label", "label_id", "identity_terms", "has_identity_term"]])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=LABEL_TO_ID["hateful"], zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_hateful": precision,
        "recall_hateful": recall,
        "f1_hateful": f1,
        "macro_f1": macro_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Train on small subsets for a smoke test.")
    parser.add_argument("--quick-size", type=int, default=128, help="Number of examples per split used with --quick.")
    parser.add_argument("--run-tag", default=None, help="Optional run folder name. Defaults to the next runN.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached model/tokenizer files only.")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = next_run_dir(base_output_dir, args.run_tag)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving this run to: {output_dir}")

    dataset = DatasetDict(
        {
            "train": load_split(data_dir / "hatexplain_train_binary.csv"),
            "validation": load_split(data_dir / "hatexplain_validation_binary.csv"),
            "test": load_split(data_dir / "hatexplain_test_binary.csv"),
        }
    )
    if args.quick:
        dataset["train"] = dataset["train"].select(range(min(args.quick_size, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(args.quick_size, len(dataset["validation"]))))
        dataset["test"] = dataset["test"].select(range(min(args.quick_size, len(dataset["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    keep_columns = {"input_ids", "attention_mask", "labels"}
    for split in tokenized:
        remove_columns = [column for column in tokenized[split].column_names if column not in keep_columns]
        tokenized[split] = tokenized[split].remove_columns(remove_columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=args.local_files_only,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")

    validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    predictions = trainer.predict(tokenized["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "confusion_matrix_test_labels_non_hateful_hateful": cm.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
