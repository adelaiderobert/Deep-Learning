from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


LABEL_TO_ID = {"non_hateful": 0, "hateful": 1}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def load_validation_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
    df["has_identity_term"] = df["has_identity_term"].astype(bool)
    return df


def predict_dataframe(df: pd.DataFrame, model_dir: Path, batch_size: int, max_length: int) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    dataset = Dataset.from_pandas(df[["id", "text", "label_id"]])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    keep_columns = {"input_ids", "attention_mask", "label_id"}
    remove_columns = [column for column in tokenized.column_names if column not in keep_columns]
    tokenized = tokenized.remove_columns(remove_columns)
    tokenized = tokenized.rename_column("label_id", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = torch.utils.data.DataLoader(tokenized, batch_size=batch_size, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Baseline run directory, e.g. outputs/baseline/run3.")
    parser.add_argument("--validation-file", default="data/processed/hatexplain_validation_binary.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "best_model"
    validation_file = Path(args.validation_file)

    df = load_validation_data(validation_file)
    predictions = predict_dataframe(df, model_dir, args.batch_size, args.max_length)

    false_positives = predictions[
        (predictions["gold_label"] == "non_hateful") & (predictions["pred_label"] == "hateful")
    ].copy()
    identity_false_positives = false_positives[false_positives["has_identity_term"]].copy()

    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    false_positives.to_csv(run_dir / "validation_false_positives.csv", index=False)
    identity_false_positives.to_csv(run_dir / "validation_identity_false_positives.csv", index=False)

    print(f"Saved validation predictions: {len(predictions)} rows")
    print(f"Saved false positives: {len(false_positives)} rows")
    print(f"Saved identity false positives: {len(identity_false_positives)} rows")


if __name__ == "__main__":
    main()
