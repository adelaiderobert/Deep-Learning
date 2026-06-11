from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from src.attribute_false_positives import attribute_text, choose_device, merge_tokens_to_words
    from src.inspect_hatexplain import IDENTITY_TERMS
except ModuleNotFoundError:
    from attribute_false_positives import attribute_text, choose_device, merge_tokens_to_words
    from inspect_hatexplain import IDENTITY_TERMS


MODELS = {
    "baseline": Path("outputs/baseline/run1"),
    "balanced_guardrail": Path("outputs/contrastive/balanced_for_attribution_m18_a0p2_b0p05"),
    "aggressive_reduction": Path("outputs/contrastive/aggressive_for_attribution_m22_a2_b0p5"),
}

CANONICAL_TERM_ALIASES = {
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

STOPWORDS = {
    "the", "and", "you", "are", "to", "of", "a", "in", "is", "it", "that", "for", "on", "with",
    "this", "they", "be", "have", "not", "as", "we", "i", "he", "she", "was", "were", "do", "but",
    "from", "or", "at", "by", "an", "if", "so", "all", "your", "their", "them", "our", "what",
    "user", "number", "url", "rt",
}


def split_terms(value) -> list[str]:
    if pd.isna(value):
        return []
    return [term.strip().lower() for term in str(value).split(",") if term.strip()]


def expanded_identity_terms(candidate_path: Path) -> list[str]:
    terms = {term.lower() for term in IDENTITY_TERMS}
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
        for column in ["matched_final_identity_terms", "identity_terms", "surface_trigger"]:
            if column in candidates.columns:
                for value in candidates[column].dropna():
                    terms.update(split_terms(value))
    return sorted(terms)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "test_predictions.csv")
    if "gold_label" not in frame.columns and "label" in frame.columns:
        frame["gold_label"] = frame["label"]
    return frame


def identity_false_positive_ids(frame: pd.DataFrame) -> set[str]:
    mask = (
        (frame["gold_label"] == "non_hateful")
        & (frame["pred_label"] == "hateful")
        & frame["has_identity_term"].astype(bool)
    )
    return set(frame.loc[mask, "id"].astype(str))


def word_matches_term(word: str, terms: list[str]) -> bool:
    word = str(word).lower().strip()
    return bool(word) and any(word == term or term in word or word in term for term in terms)


def attribution_summary(text, model, tokenizer, device, terms, n_steps: int, top_k: int) -> dict:
    scored_tokens, _ = attribute_text(text, model, tokenizer, device, 128, n_steps)
    scored_words = merge_tokens_to_words(scored_tokens)
    total_abs = sum(abs(float(item["score"])) for item in scored_words)
    identity_items = [item for item in scored_words if word_matches_term(item["word"], terms)]
    identity_abs = sum(abs(float(item["score"])) for item in identity_items)
    top_words = sorted(scored_words, key=lambda item: abs(float(item["score"])), reverse=True)[:top_k]
    return {
        "identity_abs_attribution": identity_abs,
        "total_abs_attribution": total_abs,
        "identity_attribution_share": identity_abs / total_abs if total_abs else np.nan,
        "matched_identity_words": ";".join(f"{item['word']}:{float(item['score']):.6f}" for item in identity_items),
        "top_words": ";".join(f"{item['word']}:{float(item['score']):.6f}" for item in top_words),
    }


def run_attribution(
    model_name: str,
    run_dir: Path,
    input_frame: pd.DataFrame,
    output_path: Path,
    terms: list[str],
    device: torch.device,
    n_steps: int,
    top_k: int,
    max_examples: int | None,
) -> None:
    if max_examples is not None:
        input_frame = input_frame.head(max_examples)
    model_dir = run_dir / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    rows = []
    for _, row in input_frame.iterrows():
        rows.append(
            {
                "model": model_name,
                "id": row["id"],
                "text": row["text"],
                "gold_label": row["gold_label"],
                "pred_label": row["pred_label"],
                "prob_hateful": row.get("prob_hateful", np.nan),
                **attribution_summary(row["text"], model, tokenizer, device, terms, n_steps, top_k),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_top_words(value) -> list[tuple[str, float]]:
    parsed = []
    for piece in str(value).split(";"):
        if ":" not in piece:
            continue
        word, score = piece.rsplit(":", 1)
        try:
            parsed.append((word.strip(), float(score)))
        except ValueError:
            pass
    return parsed


def canonical_map(terms: list[str]) -> dict[str, str]:
    mapping = {
        alias: canonical
        for canonical, aliases in CANONICAL_TERM_ALIASES.items()
        for alias in aliases
    }
    for term in terms:
        if len(term) >= 3 and term not in mapping:
            mapping[term] = term
    return mapping


def canonical_term(word: str, mapping: dict[str, str]) -> str | None:
    word = re.sub(r"[^a-z0-9<>]+", "", str(word).lower().strip())
    if len(word) < 3 or word in STOPWORDS:
        return None
    if word in mapping:
        return mapping[word]
    for alias, canonical in mapping.items():
        if len(alias) >= 4 and len(word) >= 4 and (alias in word or word in alias):
            return canonical
    return None


def summarize_cached_attributions(output_dir: Path, terms: list[str]) -> None:
    files = {
        name: output_dir / f"{name}_common_all_models_attribution.csv"
        for name in MODELS
    }
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing attribution files: {missing}")

    frames = [pd.read_csv(path) for path in files.values()]
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby("model", as_index=False)
        .agg(
            examples=("id", "nunique"),
            mean_identity_share=("identity_attribution_share", "mean"),
            median_identity_share=("identity_attribution_share", "median"),
            std_identity_share=("identity_attribution_share", "std"),
            mean_identity_abs=("identity_abs_attribution", "mean"),
            mean_total_abs=("total_abs_attribution", "mean"),
        )
    )
    summary.to_csv(output_dir / "identity_attribution_share_summary.csv", index=False)

    mapping = canonical_map(terms)
    rows = []
    for model_name, path in files.items():
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            scores = Counter()
            for word, score in parse_top_words(row.get("top_words", "")):
                term = canonical_term(word, mapping)
                if term is not None:
                    scores[term] += abs(score)
            total_abs = float(row.get("total_abs_attribution", np.nan))
            for term, abs_score in scores.items():
                rows.append(
                    {
                        "model": model_name,
                        "id": row["id"],
                        "term": term,
                        "term_abs_attribution_top_words": abs_score,
                        "total_abs_attribution": total_abs,
                        "term_attribution_share_top_words": abs_score / total_abs if total_abs else np.nan,
                    }
                )

    long = pd.DataFrame(rows)
    long.to_csv(output_dir / "per_term_identity_attribution_long.csv", index=False)
    per_term = (
        long.groupby(["model", "term"], as_index=False)
        .agg(
            examples_with_term=("id", "nunique"),
            mean_term_share=("term_attribution_share_top_words", "mean"),
            median_term_share=("term_attribution_share_top_words", "median"),
            mean_abs_attribution=("term_abs_attribution_top_words", "mean"),
        )
    )
    per_term.to_csv(output_dir / "per_term_identity_attribution_summary.csv", index=False)

    wide = per_term.pivot(index="term", columns="model", values="mean_term_share")
    counts = per_term.pivot(index="term", columns="model", values="examples_with_term").add_prefix("n_")
    for model_name in MODELS:
        if model_name not in wide.columns:
            wide[model_name] = np.nan
    wide["balanced_delta_vs_baseline"] = wide["balanced_guardrail"] - wide["baseline"]
    wide["aggressive_delta_vs_baseline"] = wide["aggressive_reduction"] - wide["baseline"]
    wide.join(counts, how="left").reset_index().to_csv(
        output_dir / "per_term_identity_attribution_delta_vs_baseline.csv",
        index=False,
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/attribution_comparison")
    parser.add_argument("--candidate-file", default="data/contrastive/false_positives_for_manual_pairs.csv")
    parser.add_argument("--run-attribution", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="Overwrite existing summary CSVs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "identity_attribution_share_summary.csv"
    if summary_path.exists() and not args.force and not args.run_attribution:
        print(f"Using existing attribution summary: {summary_path}")
        return
    if args.run_attribution and not args.force:
        raise ValueError("--run-attribution requires --force because it overwrites attribution CSVs.")

    terms = expanded_identity_terms(Path(args.candidate_file))
    predictions = {name: load_predictions(path) for name, path in MODELS.items()}
    fp_sets = {name: identity_false_positive_ids(frame) for name, frame in predictions.items()}
    common_ids = set.intersection(*fp_sets.values())
    common_baseline = predictions["baseline"][
        predictions["baseline"]["id"].astype(str).isin(common_ids)
    ].copy()
    common_baseline.to_csv(output_dir / "common_identity_fp_all_selected_models.csv", index=False)

    if args.run_attribution:
        device = choose_device(args.device)
        for name, run_dir in MODELS.items():
            frame = predictions[name][predictions[name]["id"].astype(str).isin(common_ids)].copy()
            run_attribution(
                name,
                run_dir,
                frame,
                output_dir / f"{name}_common_all_models_attribution.csv",
                terms,
                device,
                args.n_steps,
                args.top_k,
                args.max_examples,
            )

    summarize_cached_attributions(output_dir, terms)


if __name__ == "__main__":
    main()
