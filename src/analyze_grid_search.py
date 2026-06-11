from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


GRID_RENAME = {
    "training_margin": "margin",
    "training_alpha": "alpha",
    "training_beta": "beta",
    "contrastive_mean_loss": "loss_contrastive_mean",
    "contrastive_final_loss": "loss_contrastive_final",
    "pair_distance_final": "mean_pair_distance_final",
    "test_test_macro_f1": "test_macro_f1",
    "test_test_precision_hateful": "test_precision_hateful",
    "test_test_recall_hateful": "test_recall_hateful",
    "false_positive_counts_test_false_positives": "test_false_positives",
    "false_positive_counts_test_identity_false_positives": "test_identity_false_positives",
}

GRID_COLUMNS = [
    "source",
    "run_name",
    "margin",
    "alpha",
    "beta",
    "run_index",
    "contrastive_active_steps",
    "loss_contrastive_mean",
    "loss_contrastive_final",
    "mean_pair_distance_final",
    "test_macro_f1",
    "test_precision_hateful",
    "test_recall_hateful",
    "test_false_positives",
    "test_identity_false_positives",
]

NUMERIC_COLUMNS = [column for column in GRID_COLUMNS if column not in {"source", "run_name"}]


def standardize_grid(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.copy()
    for old, new in GRID_RENAME.items():
        if old in frame.columns and new in frame.columns:
            frame = frame.drop(columns=old)
    frame = frame.rename(columns={old: new for old, new in GRID_RENAME.items() if old in frame.columns})
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame["source"] = source
    if "run_index" not in frame.columns:
        frame["run_index"] = 1
    for column in GRID_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[GRID_COLUMNS]


def infer_run_index(run_name: str) -> int:
    match = re.search(r"run(\d+)$", str(run_name))
    return int(match.group(1)) if match else 1


def summarize_local_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists() or "smoke" in run_dir.name.lower():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training = metrics.get("training", {})
    test = metrics.get("test", {})
    fp_counts = metrics.get("false_positive_counts", {})
    confusion_matrix = metrics.get("confusion_matrix_test_labels_non_hateful_hateful")
    total_fp = fp_counts.get("test_false_positives")
    if total_fp is None and confusion_matrix is not None:
        total_fp = confusion_matrix[0][1]

    active_steps = np.nan
    loss_mean = np.nan
    loss_final = np.nan
    distance_final = np.nan
    log_path = run_dir / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        if "loss_contrastive" in log.columns and not log.empty:
            active_steps = int((log["loss_contrastive"] > 0).sum())
            loss_mean = float(log["loss_contrastive"].mean())
            loss_final = float(log["loss_contrastive"].iloc[-1])
        if "mean_pair_distance" in log.columns and not log.empty:
            distance_final = float(log["mean_pair_distance"].iloc[-1])

    return {
        "source": "local_run_scan",
        "run_name": run_dir.name,
        "margin": training.get("margin"),
        "alpha": training.get("alpha"),
        "beta": training.get("beta"),
        "run_index": infer_run_index(run_dir.name),
        "contrastive_active_steps": active_steps,
        "loss_contrastive_mean": loss_mean,
        "loss_contrastive_final": loss_final,
        "mean_pair_distance_final": distance_final,
        "test_macro_f1": test.get("test_macro_f1"),
        "test_precision_hateful": test.get("test_precision_hateful"),
        "test_recall_hateful": test.get("test_recall_hateful"),
        "test_false_positives": total_fp,
        "test_identity_false_positives": fp_counts.get("test_identity_false_positives"),
    }


def baseline_row(run_dir: Path, identity_fp_fallback: int) -> pd.DataFrame:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    confusion_matrix = metrics.get("confusion_matrix_test_labels_non_hateful_hateful")
    total_fp = confusion_matrix[0][1] if confusion_matrix is not None else np.nan
    identity_fp_path = run_dir / "test_identity_false_positives.csv"
    identity_fp = len(pd.read_csv(identity_fp_path)) if identity_fp_path.exists() else identity_fp_fallback

    return pd.DataFrame(
        [
            {
                "source": "baseline_run",
                "run_name": "baseline_run1",
                "margin": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "run_index": 1,
                "contrastive_active_steps": 0,
                "loss_contrastive_mean": 0.0,
                "loss_contrastive_final": 0.0,
                "mean_pair_distance_final": np.nan,
                "test_macro_f1": test["test_macro_f1"],
                "test_precision_hateful": test["test_precision_hateful"],
                "test_recall_hateful": test["test_recall_hateful"],
                "test_false_positives": total_fp,
                "test_identity_false_positives": identity_fp,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", default="outputs/baseline/run1")
    parser.add_argument("--contrastive-dir", default="outputs/contrastive")
    parser.add_argument("--identity-fp-fallback", type=int, default=26)
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis CSVs.")
    args = parser.parse_args()

    contrastive_dir = Path(args.contrastive_dir)
    output_path = contrastive_dir / "all_model_comparison_clean.csv"
    if output_path.exists() and not args.force:
        print(f"Using existing comparison table: {output_path}")
        return

    baseline = baseline_row(Path(args.baseline_run_dir), args.identity_fp_fallback)
    frames = [baseline]

    summary_files = [
        ("focused_margin_grid_summary.csv", "focused_margin_grid"),
        ("big_margin_grid_summary.csv", "big_margin_grid"),
    ]
    for filename, source in summary_files:
        path = contrastive_dir / filename
        if path.exists():
            frames.append(standardize_grid(pd.read_csv(path), source))

    local_rows = []
    for run_dir in sorted(contrastive_dir.iterdir()):
        if run_dir.is_dir():
            row = summarize_local_run(run_dir)
            if row is not None:
                local_rows.append(row)
    if local_rows:
        local = standardize_grid(pd.DataFrame(local_rows), "local_run_scan")
        local.to_csv(contrastive_dir / "local_contrastive_run_summary.csv", index=False)
        frames.append(local)

    raw = pd.concat(frames, ignore_index=True)
    source_rank = {"local_run_scan": 3, "big_margin_grid": 2, "focused_margin_grid": 1}
    contrastive = raw[raw["source"] != "baseline_run"].copy()
    contrastive["source_rank"] = contrastive["source"].map(source_rank).fillna(0)
    contrastive = (
        contrastive.sort_values(["source_rank", "run_index"])
        .drop_duplicates(["margin", "alpha", "beta"], keep="last")
        .drop(columns="source_rank")
    )

    all_models = pd.concat([baseline, contrastive], ignore_index=True)
    all_models.insert(
        0,
        "model_family",
        np.where(all_models["source"] == "baseline_run", "baseline", "contrastive_mixed_loss"),
    )
    baseline_values = baseline.iloc[0]
    for metric, output_name in [
        ("test_macro_f1", "delta_macro_f1_vs_baseline"),
        ("test_recall_hateful", "delta_recall_vs_baseline"),
        ("test_precision_hateful", "delta_precision_vs_baseline"),
        ("test_false_positives", "delta_total_fp_vs_baseline"),
        ("test_identity_false_positives", "delta_identity_fp_vs_baseline"),
    ]:
        all_models[output_name] = all_models[metric] - baseline_values[metric]

    all_models.to_csv(output_path, index=False)
    print(f"Saved {len(all_models)} model rows to {output_path}")


if __name__ == "__main__":
    main()
