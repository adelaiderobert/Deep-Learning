from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed


LABEL_TO_ID = {"non_hateful": 0, "hateful": 1}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def format_float_for_tag(value: float) -> str:
    """Format a hyperparameter value so it is readable and filesystem-safe."""
    formatted = f"{value:g}".replace("-", "m").replace(".", "p")
    return formatted


def contrastive_run_prefix(alpha: float, beta: float, margin: float, quick: bool) -> str:
    """Create the default run-folder prefix from the main loss weights."""
    prefix = (
        f"alpha{format_float_for_tag(alpha)}"
        f"_beta{format_float_for_tag(beta)}"
        f"_margin{format_float_for_tag(margin)}"
    )
    return f"smoke_{prefix}" if quick else prefix


def next_run_dir(base_output_dir: Path, run_tag: str | None = None, run_prefix: str | None = None) -> Path:
    if run_tag:
        run_dir = base_output_dir / run_tag
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        return run_dir

    if run_prefix:
        run_idx = 1
        while (base_output_dir / f"{run_prefix}_run{run_idx}").exists():
            run_idx += 1
        return base_output_dir / f"{run_prefix}_run{run_idx}"

    run_idx = 1
    while (base_output_dir / f"run{run_idx}").exists():
        run_idx += 1
    return base_output_dir / f"run{run_idx}"


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_binary_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    df["has_identity_term"] = df["has_identity_term"].astype(bool)
    return df


def tokenize_dataframe(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df[["id", "text", "label", "label_id", "identity_terms", "has_identity_term"]])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    keep_columns = {"input_ids", "attention_mask", "labels"}
    remove_columns = [column for column in tokenized.column_names if column not in keep_columns]
    return tokenized.remove_columns(remove_columns)


class ContrastivePairDataset(TorchDataset):
    def __init__(self, pair_df: pd.DataFrame, non_hateful_column: str, hateful_column: str) -> None:
        self.non_hateful_texts = pair_df[non_hateful_column].astype(str).tolist()
        self.hateful_texts = pair_df[hateful_column].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.non_hateful_texts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "non_hateful_text": self.non_hateful_texts[idx],
            "hateful_text": self.hateful_texts[idx],
        }


def load_contrastive_pairs(path: Path, non_hateful_column: str, hateful_column: str) -> pd.DataFrame:
    pair_df = pd.read_csv(path)
    missing_columns = [column for column in [non_hateful_column, hateful_column] if column not in pair_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required pair columns in {path}: {missing_columns}")

    pair_df = pair_df.dropna(subset=[non_hateful_column, hateful_column]).copy()
    pair_df[non_hateful_column] = pair_df[non_hateful_column].astype(str).str.strip()
    pair_df[hateful_column] = pair_df[hateful_column].astype(str).str.strip()
    pair_df = pair_df[(pair_df[non_hateful_column] != "") & (pair_df[hateful_column] != "")]

    if pair_df.empty:
        raise ValueError(
            f"No completed contrastive pairs found in {path}. "
            f"Fill the '{hateful_column}' column before training."
        )
    return pair_df


def compute_metrics_from_arrays(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=LABEL_TO_ID["hateful"],
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_hateful": precision,
        "recall_hateful": recall,
        "f1_hateful": f1,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> pd.DataFrame:
    tokenized = tokenize_dataframe(df, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collator)

    model.eval()
    probabilities = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(np.argmax(probs, axis=-1))

    probs_array = np.concatenate(probabilities, axis=0)
    pred_array = np.concatenate(predictions, axis=0)

    output = df.copy()
    output["gold_label"] = output["label"]
    output["pred_label"] = [ID_TO_LABEL[int(pred)] for pred in pred_array]
    output["prob_non_hateful"] = probs_array[:, LABEL_TO_ID["non_hateful"]]
    output["prob_hateful"] = probs_array[:, LABEL_TO_ID["hateful"]]
    return output


def evaluate_model(
    model,
    tokenizer,
    df: pd.DataFrame,
    split_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = predict_dataframe(model, tokenizer, df, batch_size, max_length, device)
    labels = predictions["label"].map(LABEL_TO_ID).to_numpy()
    preds = predictions["pred_label"].map(LABEL_TO_ID).to_numpy()
    metrics = {f"{split_name}_{key}": value for key, value in compute_metrics_from_arrays(labels, preds).items()}
    return metrics, predictions


def false_positive_slices(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    false_positives = predictions[
        (predictions["gold_label"] == "non_hateful") & (predictions["pred_label"] == "hateful")
    ].copy()
    identity_false_positives = false_positives[false_positives["has_identity_term"]].copy()
    return false_positives, identity_false_positives


def pair_collate_fn(batch, tokenizer, max_length: int) -> dict[str, dict[str, torch.Tensor]]:
    non_hateful_texts = [item["non_hateful_text"] for item in batch]
    hateful_texts = [item["hateful_text"] for item in batch]
    non_hateful_inputs = tokenizer(
        non_hateful_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    hateful_inputs = tokenizer(
        hateful_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    non_hateful_inputs["labels"] = torch.zeros(len(batch), dtype=torch.long)
    hateful_inputs["labels"] = torch.ones(len(batch), dtype=torch.long)
    return {"non_hateful": non_hateful_inputs, "hateful": hateful_inputs}


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--pair-file", default="data/contrastive/false_positives_for_manual_pairs.csv")
    parser.add_argument("--output-dir", default="outputs/contrastive")
    parser.add_argument("--init-model-dir", required=True, help="Baseline checkpoint folder, e.g. outputs/baseline/run8/best_model.")
    parser.add_argument("--non-hateful-column", default="non_hateful_text")
    parser.add_argument("--hateful-column", default="hateful_counterpart")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for the contrastive pair loss.")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for cross-entropy on pair sentences.")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-size", type=int, default=128)
    parser.add_argument("--quick-pairs", type=int, default=16)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)

    data_dir = Path(args.data_dir)
    pair_file = Path(args.pair_file)
    init_model_dir = Path(args.init_model_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = contrastive_run_prefix(args.alpha, args.beta, args.margin, args.quick)
    output_dir = next_run_dir(base_output_dir, args.run_tag, run_prefix)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving this run to: {output_dir}")
    print(f"Using device: {device}")

    train_df = load_binary_split(data_dir / "hatexplain_train_binary.csv")
    validation_df = load_binary_split(data_dir / "hatexplain_validation_binary.csv")
    test_df = load_binary_split(data_dir / "hatexplain_test_binary.csv")
    pair_df = load_contrastive_pairs(pair_file, args.non_hateful_column, args.hateful_column)

    if args.quick:
        train_df = train_df.head(args.quick_size).copy()
        validation_df = validation_df.head(args.quick_size).copy()
        test_df = test_df.head(args.quick_size).copy()
        pair_df = pair_df.head(args.quick_pairs).copy()

    tokenizer = AutoTokenizer.from_pretrained(init_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_model_dir,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=True,
    )
    model.to(device)

    train_dataset = tokenize_dataframe(train_df, tokenizer, args.max_length)
    train_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collator)

    pair_dataset = ContrastivePairDataset(pair_df, args.non_hateful_column, args.hateful_column)
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=args.pair_batch_size,
        shuffle=True,
        collate_fn=lambda batch: pair_collate_fn(batch, tokenizer, args.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    train_iterator = cycle(train_loader)
    pair_iterator = cycle(pair_loader)

    training_log = []
    progress = tqdm(range(total_steps), desc="contrastive training")
    for step in progress:
        model.train()
        train_batch = batch_to_device(next(train_iterator), device)
        pair_batch = next(pair_iterator)
        pair_non_hateful = batch_to_device(pair_batch["non_hateful"], device)
        pair_hateful = batch_to_device(pair_batch["hateful"], device)

        train_outputs = model(**train_batch)
        loss_train_ce = train_outputs.loss

        non_outputs = model(**pair_non_hateful, output_hidden_states=True)
        hate_outputs = model(**pair_hateful, output_hidden_states=True)
        loss_pair_ce = 0.5 * (non_outputs.loss + hate_outputs.loss)

        non_embeddings = non_outputs.hidden_states[-1][:, 0, :]
        hate_embeddings = hate_outputs.hidden_states[-1][:, 0, :]
        distances = torch.norm(non_embeddings - hate_embeddings, p=2, dim=1)
        loss_contrastive = torch.relu(args.margin - distances).pow(2).mean()

        loss = loss_train_ce + args.beta * loss_pair_ce + args.alpha * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_row = {
            "step": step + 1,
            "loss_total": float(loss.detach().cpu()),
            "loss_train_ce": float(loss_train_ce.detach().cpu()),
            "loss_pair_ce": float(loss_pair_ce.detach().cpu()),
            "loss_contrastive": float(loss_contrastive.detach().cpu()),
            "mean_pair_distance": float(distances.detach().mean().cpu()),
        }
        training_log.append(log_row)
        progress.set_postfix({key: f"{value:.4f}" for key, value in log_row.items() if key != "step"})

    model.save_pretrained(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    pair_df.to_csv(output_dir / "contrastive_pairs_used.csv", index=False)
    train_df.to_csv(output_dir / "training_data_used.csv", index=False)
    pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)

    validation_metrics, validation_predictions = evaluate_model(
        model, tokenizer, validation_df, "validation", args.batch_size, args.max_length, device
    )
    test_metrics, test_predictions = evaluate_model(
        model, tokenizer, test_df, "test", args.batch_size, args.max_length, device
    )

    validation_false_positives, validation_identity_false_positives = false_positive_slices(validation_predictions)
    test_false_positives, test_identity_false_positives = false_positive_slices(test_predictions)

    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    validation_false_positives.to_csv(output_dir / "validation_false_positives.csv", index=False)
    validation_identity_false_positives.to_csv(output_dir / "validation_identity_false_positives.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    test_false_positives.to_csv(output_dir / "test_false_positives.csv", index=False)
    test_identity_false_positives.to_csv(output_dir / "test_identity_false_positives.csv", index=False)

    y_true = test_predictions["label"].map(LABEL_TO_ID).to_numpy()
    y_pred = test_predictions["pred_label"].map(LABEL_TO_ID).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "training": {
            "init_model_dir": str(init_model_dir),
            "pair_file": str(pair_file),
            "num_train_examples": len(train_df),
            "num_contrastive_pairs": len(pair_df),
            "epochs": args.epochs,
            "total_steps": total_steps,
            "alpha": args.alpha,
            "beta": args.beta,
            "margin": args.margin,
            "learning_rate": args.learning_rate,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "confusion_matrix_test_labels_non_hateful_hateful": cm.tolist(),
        "false_positive_counts": {
            "validation_false_positives": len(validation_false_positives),
            "validation_identity_false_positives": len(validation_identity_false_positives),
            "test_false_positives": len(test_false_positives),
            "test_identity_false_positives": len(test_identity_false_positives),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
