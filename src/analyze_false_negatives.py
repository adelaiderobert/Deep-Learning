from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "baseline": Path("outputs/baseline/run1"),
    "balanced_guardrail": Path("outputs/contrastive/balanced_for_attribution_m18_a0p2_b0p05"),
    "aggressive_reduction": Path("outputs/contrastive/aggressive_for_attribution_m22_a2_b0p5"),
}

TERM_ALIASES = {
    "muslim": {"muslim", "muslims", "moslem", "moslems", "moslim", "moslimes", "muslimes", "muzzie", "muzzies", "islam"},
    "jewish": {"jewish", "jews", "jew", "kike", "heeb"},
    "black": {"black", "blacks", "coon", "coons", "nigger", "niggers", "nigga", "negress", "negro", "dindu"},
    "white": {"white", "whites", "whigger"},
    "women": {"woman", "women", "girl", "girls", "female", "females"},
    "gay_lgbt": {"gay", "lesbian", "trans", "dyke", "dykes", "faggot", "faggots", "faggotry", "fags", "queer", "queers"},
    "immigrant_nationality": {"immigrant", "immigrants", "mexican", "asian", "arab", "afghani", "spic"},
    "disability": {"disabled", "retarded", "retard"},
    "other_trigger": {"shitskin", "cucks", "scum", "scumbag", "stupid", "cowards", "maggots"},
}

ALIAS_TO_GROUP = {
    alias: group
    for group, aliases in TERM_ALIASES.items()
    for alias in aliases
}


def term_groups_in_text(text: str) -> list[str]:
    groups = set()
    for token in re.findall(r"[a-z0-9<>]+", str(text).lower()):
        if token in ALIAS_TO_GROUP:
            groups.add(ALIAS_TO_GROUP[token])
            continue
        for alias, group in ALIAS_TO_GROUP.items():
            if len(alias) >= 5 and len(token) >= 5 and (alias in token or token in alias):
                groups.add(group)
    return sorted(groups)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "test_predictions.csv")
    if "gold_label" not in frame.columns and "label" in frame.columns:
        frame["gold_label"] = frame["label"]
    required = {"id", "text", "gold_label", "pred_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{run_dir} predictions are missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/false_negative_analysis")
    parser.add_argument(
        "--attribution-delta-file",
        default="outputs/attribution_comparison/per_term_identity_attribution_delta_vs_baseline.csv",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis CSVs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "false_negative_summary_by_model.csv"
    if summary_path.exists() and not args.force:
        print(f"Using existing false-negative summary: {summary_path}")
        return

    summary_rows = []
    per_term_rows = []
    example_rows = []

    for model_name, run_dir in MODELS.items():
        predictions = load_predictions(run_dir)
        predictions["term_groups"] = predictions["text"].apply(term_groups_in_text)
        predictions["has_term_group"] = predictions["term_groups"].apply(bool)
        predictions["is_false_negative"] = (
            (predictions["gold_label"] == "hateful")
            & (predictions["pred_label"] == "non_hateful")
        )

        hateful = predictions[predictions["gold_label"] == "hateful"]
        hateful_identity = hateful[hateful["has_term_group"]]
        false_negatives = hateful[hateful["is_false_negative"]]
        identity_false_negatives = false_negatives[false_negatives["has_term_group"]]
        summary_rows.append(
            {
                "model": model_name,
                "total_hateful_examples": len(hateful),
                "total_false_negatives": len(false_negatives),
                "total_hateful_recall": 1 - len(false_negatives) / len(hateful),
                "identity_hateful_examples": len(hateful_identity),
                "identity_false_negatives": len(identity_false_negatives),
                "identity_hateful_recall": 1 - len(identity_false_negatives) / len(hateful_identity),
            }
        )

        for _, row in identity_false_negatives.iterrows():
            example_rows.append(
                {
                    "model": model_name,
                    "id": row["id"],
                    "text": row["text"],
                    "term_groups": ",".join(row["term_groups"]),
                    "prob_hateful": row.get("prob_hateful", np.nan),
                }
            )

        observed_groups = sorted({group for groups in hateful["term_groups"] for group in groups})
        for group in observed_groups:
            group_hateful = hateful[hateful["term_groups"].apply(lambda groups, g=group: g in groups)]
            group_false_negatives = group_hateful[group_hateful["is_false_negative"]]
            per_term_rows.append(
                {
                    "model": model_name,
                    "term_group": group,
                    "hateful_examples_with_term": len(group_hateful),
                    "false_negatives_with_term": len(group_false_negatives),
                    "term_recall": 1 - len(group_false_negatives) / len(group_hateful),
                    "mean_prob_hateful": group_hateful["prob_hateful"].mean(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    per_term = pd.DataFrame(per_term_rows)
    examples = pd.DataFrame(example_rows)
    summary.to_csv(output_dir / "false_negative_summary_by_model.csv", index=False)
    per_term.to_csv(output_dir / "false_negative_recall_by_term.csv", index=False)
    examples.to_csv(output_dir / "identity_false_negative_examples.csv", index=False)

    recall = per_term.pivot(index="term_group", columns="model", values="term_recall").add_suffix("_recall")
    counts = per_term.pivot(index="term_group", columns="model", values="hateful_examples_with_term").add_prefix("n_")
    false_negatives = per_term.pivot(index="term_group", columns="model", values="false_negatives_with_term").add_prefix("fn_")
    probabilities = per_term.pivot(index="term_group", columns="model", values="mean_prob_hateful").add_suffix("_mean_prob_hateful")
    comparison = recall.join([counts, false_negatives, probabilities], how="outer").reset_index()
    comparison["balanced_delta_recall_vs_baseline"] = comparison["balanced_guardrail_recall"] - comparison["baseline_recall"]
    comparison["aggressive_delta_recall_vs_baseline"] = comparison["aggressive_reduction_recall"] - comparison["baseline_recall"]
    comparison.to_csv(output_dir / "term_recall_delta_vs_baseline.csv", index=False)

    attribution_path = Path(args.attribution_delta_file)
    if attribution_path.exists():
        combined = comparison.merge(
            pd.read_csv(attribution_path),
            left_on="term_group",
            right_on="term",
            how="left",
        )
        combined.to_csv(output_dir / "term_recall_vs_attribution_delta.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
