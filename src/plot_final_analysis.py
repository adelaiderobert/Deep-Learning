from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["baseline", "balanced_guardrail", "aggressive_reduction"]


def plot_grid_search(comparison_path: Path, output_dir: Path) -> None:
    models = pd.read_csv(comparison_path)
    baseline = models[models["source"] == "baseline_run"].iloc[0]
    contrastive = models[models["source"] != "baseline_run"].dropna(subset=["margin"]).copy()
    contrastive["margin_label"] = contrastive["margin"].astype(int).astype(str)
    margin_order = [str(int(value)) for value in sorted(contrastive["margin"].unique())]

    balanced = contrastive[
        (contrastive["margin"] == 18)
        & (contrastive["alpha"] == 0.2)
        & (contrastive["beta"] == 0.05)
    ].iloc[0]
    aggressive = contrastive[
        (contrastive["margin"] == 22)
        & (contrastive["alpha"] == 2.0)
        & (contrastive["beta"] == 0.5)
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    palette = dict(zip(margin_order, sns.color_palette("viridis", len(margin_order))))
    for axis, x_column, title in [
        (axes[0], "test_identity_false_positives", "Identity false positives vs recall"),
        (axes[1], "test_false_positives", "Total false positives vs recall"),
    ]:
        sns.scatterplot(
            data=contrastive,
            x=x_column,
            y="test_recall_hateful",
            hue="margin_label",
            hue_order=margin_order,
            size="test_macro_f1",
            sizes=(35, 280),
            palette=palette,
            alpha=0.65,
            ax=axis,
        )
        axis.axvline(baseline[x_column], color="black", linestyle="--", linewidth=1.2)
        axis.axhline(baseline["test_recall_hateful"], color="black", linestyle="--", linewidth=1.2)
        for row, label, marker, color in [
            (baseline, "Baseline", "*", "black"),
            (balanced, "Balanced", "D", "#2ca25f"),
            (aggressive, "Aggressive", "D", "#de2d26"),
        ]:
            axis.scatter(row[x_column], row["test_recall_hateful"], s=420, marker=marker, color=color, edgecolor="white", zorder=10)
            axis.annotate(label, (row[x_column], row["test_recall_hateful"]), xytext=(7, 7), textcoords="offset points")
        axis.set_title(title)
        axis.set_xlabel(f"{x_column.replace('_', ' ')} (lower is better)")
        axis.set_ylabel("Hateful recall (higher is better)")
    fig.tight_layout()
    fig.savefig(output_dir / "final_tradeoff_scatter_with_margin22.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    margins = [value for value in [15, 18, 20, 22] if value in set(contrastive["margin"])]
    for metric, filename, cmap, fmt in [
        ("test_identity_false_positives", "identity_fp_heatmaps.png", "Blues_r", ".0f"),
        ("test_macro_f1", "macro_f1_heatmaps.png", "YlGnBu", ".3f"),
    ]:
        fig, axes = plt.subplots(1, len(margins), figsize=(6 * len(margins), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for axis, margin in zip(axes, margins):
            subset = contrastive[contrastive["margin"] == margin]
            pivot = subset.pivot_table(index="beta", columns="alpha", values=metric, aggfunc="min")
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, cbar=axis is axes[-1], ax=axis)
            axis.set_title(f"{metric.replace('_', ' ')}, margin={margin:g}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_attribution(attribution_dir: Path) -> None:
    files = {
        name: attribution_dir / f"{name}_common_all_models_attribution.csv"
        for name in MODEL_ORDER
    }
    combined = pd.concat([pd.read_csv(path) for path in files.values()], ignore_index=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=combined, x="model", y="identity_attribution_share", order=MODEL_ORDER, ax=axis)
    sns.stripplot(data=combined, x="model", y="identity_attribution_share", order=MODEL_ORDER, color="black", alpha=0.45, ax=axis)
    axis.set_title("Identity-term attribution on shared false positives")
    axis.set_xlabel("Model")
    axis.set_ylabel("Identity attribution share")
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(attribution_dir / "identity_attribution_share_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_term = pd.read_csv(attribution_dir / "per_term_identity_attribution_summary.csv")
    terms = (
        per_term[per_term["model"] == "baseline"]
        .sort_values("mean_term_share", ascending=False)
        .head(12)["term"]
    )
    plot_frame = per_term[per_term["term"].isin(terms)]
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_frame, x="term", y="mean_term_share", hue="model", order=list(terms), ax=axis)
    axis.set_title("Per-term attribution share on shared false positives")
    axis.set_xlabel("Identity or trigger group")
    axis.set_ylabel("Mean attribution share")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(attribution_dir / "per_term_identity_attribution_barplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_false_negatives(false_negative_dir: Path) -> None:
    summary = pd.read_csv(false_negative_dir / "false_negative_summary_by_model.csv")
    plot_frame = summary.melt(
        id_vars="model",
        value_vars=["total_false_negatives", "identity_false_negatives"],
        var_name="false_negative_type",
        value_name="count",
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_frame, x="model", y="count", hue="false_negative_type", order=MODEL_ORDER, ax=axis)
    axis.set_title("False negatives by model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(false_negative_dir / "false_negative_counts_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_term = pd.read_csv(false_negative_dir / "false_negative_recall_by_term.csv")
    baseline_counts = per_term[per_term["model"] == "baseline"].set_index("term_group")["hateful_examples_with_term"]
    terms = baseline_counts[baseline_counts >= 2].index
    plot_frame = per_term[per_term["term_group"].isin(terms)]
    fig, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_frame, x="term_group", y="term_recall", hue="model", order=terms, hue_order=MODEL_ORDER, ax=axis)
    axis.set_title("Per-term recall on hateful identity examples")
    axis.set_xlabel("Identity or trigger group")
    axis.set_ylabel("Hateful recall")
    axis.set_ylim(0, 1.02)
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(false_negative_dir / "per_term_hateful_recall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-file", default="outputs/contrastive/all_model_comparison_clean.csv")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--attribution-dir", default="outputs/attribution_comparison")
    parser.add_argument("--false-negative-dir", default="outputs/false_negative_analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing plot files.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    main_plot = plot_dir / "final_tradeoff_scatter_with_margin22.png"
    if main_plot.exists() and not args.force:
        print(f"Using existing plots. Pass --force to regenerate them.")
        return

    plot_grid_search(Path(args.comparison_file), plot_dir)
    plot_attribution(Path(args.attribution_dir))
    plot_false_negatives(Path(args.false_negative_dir))
    print("Saved final analysis plots.")


if __name__ == "__main__":
    main()
