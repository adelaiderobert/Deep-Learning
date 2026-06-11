from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests


IDENTITY_TERMS = [
    "muslim",
    "jewish",
    "christian",
    "black",
    "white",
    "women",
    "woman",
    "gay",
    "lesbian",
    "trans",
    "immigrant",
    "mexican",
    "asian",
    "arab",
    "disabled",
]

LABEL_MAP = {
    "hatespeech": "hateful",
    "normal": "non_hateful",
}

RAW_URLS = {
    "dataset": "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json",
    "post_id_divisions": "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/post_id_divisions.json",
}

RAW_LABEL_NAMES = ["hatespeech", "normal", "offensive"]

def normalize_label(value: Any, label_names: list[str] | None = None) -> str | None:
    if isinstance(value, int) and label_names:
        value = label_names[value]
    label = str(value).lower().strip()
    return LABEL_MAP.get(label)

def normalize_raw_label(value: Any, label_names: list[str] | None = None) -> str | None:
    if isinstance(value, int) and label_names:
        value = label_names[value]
    label = str(value).lower().strip()
    if label in RAW_LABEL_NAMES:
        return label
    return None

def majority_label(example: dict[str, Any], label_names: list[str] | None = None) -> str | None:
    if "label" in example:
        raw_label = normalize_raw_label(example["label"], label_names)
        return normalize_label(raw_label, label_names) if raw_label else None

    annotators = example.get("annotators")
    if isinstance(annotators, dict) and "label" in annotators:
        labels = annotators["label"]
        raw_labels = [normalize_raw_label(label, label_names) for label in labels]
        raw_labels = [label for label in raw_labels if label is not None]
        if raw_labels:
            raw_label, count = Counter(raw_labels).most_common(1)[0]
            if count > 1:
                return normalize_label(raw_label, label_names)
    if isinstance(annotators, list):
        labels = [annotator.get("label") for annotator in annotators if isinstance(annotator, dict)]
        raw_labels = [normalize_raw_label(label, label_names) for label in labels]
        raw_labels = [label for label in raw_labels if label is not None]
        if raw_labels:
            raw_label, count = Counter(raw_labels).most_common(1)[0]
            if count > 1:
                return normalize_label(raw_label, label_names)

    return None

def example_text(example: dict[str, Any]) -> str:
    if "text" in example and example["text"]:
        return str(example["text"])
    if "post_tokens" in example and example["post_tokens"]:
        return " ".join(str(token) for token in example["post_tokens"])
    if "comment" in example and example["comment"]:
        return str(example["comment"])
    raise ValueError(f"Could not find text field in example keys: {sorted(example)}")

def identity_terms_in_text(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for term in IDENTITY_TERMS:
        if re.search(rf"\b{re.escape(term)}s?\b", lowered):
            found.append(term)
    return found

def processed_frame(dataset, label_names: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for idx, example in enumerate(dataset):
        label = majority_label(example, label_names)
        if label is None:
            continue
        text = example_text(example)
        terms = identity_terms_in_text(text)
        rows.append(
            {
                "id": example.get("post_id", idx),
                "text": text,
                "label": label,
                "identity_terms": ",".join(terms),
                "has_identity_term": bool(terms),
            }
        )
    return pd.DataFrame(rows)

def original_labels_frame(dataset) -> pd.DataFrame:
    rows = []
    for idx, example in enumerate(dataset):
        text = example_text(example)
        terms = identity_terms_in_text(text)
        annotators = example.get("annotators", [])
        row = {
            "id": example.get("post_id", example.get("id", idx)),
            "text": text,
            "identity_terms": ",".join(terms),
            "has_identity_term": bool(terms),
        }
        if isinstance(annotators, list):
            for annotator_index, annotator in enumerate(annotators, start=1):
                if not isinstance(annotator, dict):
                    continue
                row[f"annotator_{annotator_index}_id"] = annotator.get("annotator_id")
                row[f"annotator_{annotator_index}_label"] = annotator.get("label")
                row[f"annotator_{annotator_index}_target"] = ",".join(
                    str(target) for target in annotator.get("target", [])
                )
        rows.append(row)
    return pd.DataFrame(rows)

def download_json(url: str, output_path: Path) -> Any:
    if not output_path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_text(response.text, encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))

def load_hatexplain(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = download_json(RAW_URLS["dataset"], raw_dir / "hatexplain_dataset.json")
    divisions = download_json(RAW_URLS["post_id_divisions"], raw_dir / "hatexplain_post_id_divisions.json")

    return {
        "train": [{"id": post_id, **dataset[post_id]} for post_id in divisions["train"]],
        "validation": [{"id": post_id, **dataset[post_id]} for post_id in divisions["val"]],
        "test": [{"id": post_id, **dataset[post_id]} for post_id in divisions["test"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_hatexplain(raw_dir)
    print({split: len(rows) for split, rows in dataset.items()})

    summary_rows = []
    for split_name, split_dataset in dataset.items():
        original_df = original_labels_frame(split_dataset)
        original_df.to_csv(output_dir / f"hatexplain_{split_name}_original_labels.csv", index=False)

        df = processed_frame(split_dataset, RAW_LABEL_NAMES)
        df.to_csv(output_dir / f"hatexplain_{split_name}_binary.csv", index=False)

        counts = df["label"].value_counts().to_dict()
        identity_counts = (
            df.groupby("label")["has_identity_term"].sum().astype(int).to_dict()
            if not df.empty
            else {}
        )
        summary_rows.append(
            {
                "split": split_name,
                "rows_after_mapping": len(df),
                "hateful": counts.get("hateful", 0),
                "non_hateful": counts.get("non_hateful", 0),
                "identity_hateful": identity_counts.get("hateful", 0),
                "identity_non_hateful": identity_counts.get("non_hateful", 0),
            }
        )

        print(f"\n{split_name}")
        print(df["label"].value_counts())
        print("\nExamples:")
        for label in ["hateful", "non_hateful"]:
            examples = df[df["label"] == label].head(3)
            for text in examples["text"]:
                print(f"[{label}] {text[:220]}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "hatexplain_binary_summary.csv", index=False)
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
