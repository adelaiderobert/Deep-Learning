from __future__ import annotations

import argparse
import re
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

IDENTITY_ALIASES = {
    "muslim", "muslims", "moslem", "moslems", "moslim", "moslimes", "muslimes", "muzzie", "muzzies", "islam",
    "jewish", "jews", "jew", "kike", "heeb",
    "black", "blacks", "coon", "coons", "nigger", "niggers", "nigga", "negress", "negro", "dindu",
    "white", "whites", "whigger",
    "woman", "women", "girl", "girls", "female", "females",
    "gay", "lesbian", "trans", "dyke", "dykes", "faggot", "faggots", "faggotry", "fags", "queer", "queers",
    "immigrant", "immigrants", "mexican", "asian", "arab", "afghani", "spic",
    "disabled", "retarded", "retard",
    "shitskin", "towelhead", "chink", "chinks", "redskin", "redskins",
}

HATEFUL_CONTEXT_TERMS = {
    "hate", "hates", "hated", "hating",
    "kill", "kills", "killed", "killing", "murder", "murdered", "murdering",
    "die", "dies", "died", "death", "dead",
    "destroy", "destroys", "destroyed", "destroying", "ruin", "ruins", "ruined", "ruining",
    "remove", "removed", "removing", "deport", "deported", "deporting", "ban", "banned", "exclude", "excluded",
    "inferior", "superior", "worthless", "disgusting", "filthy", "vermin", "parasite", "parasites", "disease",
    "terrorist", "terrorists", "criminal", "criminals", "threat", "threats", "threaten", "threatening",
    "attack", "attacks", "attacked", "attacking", "violence", "violent",
    "rape", "raped", "rapist", "rapists", "abuse", "abused", "abuser",
    "genocide", "exterminate", "exterminated", "eradicate", "eradicated",
    "invasion", "invade", "invaded", "invading",
}

STOPWORDS = {
    "the", "and", "you", "are", "to", "of", "a", "in", "is", "it", "that", "for", "on", "with", "this",
    "they", "be", "have", "not", "as", "we", "i", "he", "she", "was", "were", "do", "but", "from", "or",
    "at", "by", "an", "if", "so", "all", "your", "their", "them", "our", "what", "user", "number", "url", "rt",
}


def split_terms(value) -> list[str]:
    if pd.isna(value):
        return []
    return [term.strip().lower() for term in str(value).split(",") if term.strip()]


def build_identity_vocabulary(candidate_path: Path) -> set[str]:
    terms = {term.lower() for term in IDENTITY_TERMS}
    terms.update(IDENTITY_ALIASES)
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
        for column in ["matched_final_identity_terms", "identity_terms", "surface_trigger"]:
            if column in candidates.columns:
                for value in candidates[column].dropna():
                    terms.update(split_terms(value))
    return {term for term in terms if len(term) >= 3}


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9<>]+", "", str(word).lower().strip())


def word_in_vocabulary(word: str, vocabulary: set[str]) -> bool:
    word = normalize_word(word)
    if len(word) < 3:
        return False
    if word in vocabulary:
        return True
    return any(
        len(term) >= 4 and len(word) >= 4 and (term in word or word in term)
        for term in vocabulary
    )


def classify_attribution_word(
    word: str,
    identity_vocabulary: set[str],
    context_vocabulary: set[str] = HATEFUL_CONTEXT_TERMS,
) -> str:
    clean = normalize_word(word)
    if not clean or clean in STOPWORDS:
        return "other"
    if word_in_vocabulary(clean, identity_vocabulary):
        return "identity"
    if word_in_vocabulary(clean, context_vocabulary):
        return "non_identity_hateful_context"
    return "other"


def attribution_specificity_for_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    identity_vocabulary: set[str],
    max_length: int = 128,
    n_steps: int = 16,
) -> dict:
    scored_tokens, _ = attribute_text(text, model, tokenizer, device, max_length, n_steps)
    scored_words = merge_tokens_to_words(scored_tokens)
    bucket_scores = {
        "identity": 0.0,
        "non_identity_hateful_context": 0.0,
        "other": 0.0,
    }
    bucket_words = {bucket: [] for bucket in bucket_scores}

    for item in scored_words:
        word = str(item["word"])
        score = float(item["score"])
        bucket = classify_attribution_word(word, identity_vocabulary)
        bucket_scores[bucket] += abs(score)
        bucket_words[bucket].append((word, score))

    total_abs = sum(bucket_scores.values())
    top_non_identity = [
        (str(item["word"]), float(item["score"]))
        for item in sorted(scored_words, key=lambda value: abs(float(value["score"])), reverse=True)
        if classify_attribution_word(str(item["word"]), identity_vocabulary) != "identity"
        and normalize_word(str(item["word"])) not in STOPWORDS
    ][:8]

    return {
        "total_abs_attribution": total_abs,
        "identity_abs_attribution": bucket_scores["identity"],
        "context_abs_attribution": bucket_scores["non_identity_hateful_context"],
        "other_abs_attribution": bucket_scores["other"],
        "identity_share": bucket_scores["identity"] / total_abs if total_abs else np.nan,
        "context_share": bucket_scores["non_identity_hateful_context"] / total_abs if total_abs else np.nan,
        "other_share": bucket_scores["other"] / total_abs if total_abs else np.nan,
        "top_identity_words": format_top_words(bucket_words["identity"]),
        "top_context_words": format_top_words(bucket_words["non_identity_hateful_context"]),
        "top_non_identity_content_words": ";".join(f"{word}:{score:.5f}" for word, score in top_non_identity),
    }


def format_top_words(words: list[tuple[str, float]], limit: int = 8) -> str:
    ordered = sorted(words, key=lambda value: abs(value[1]), reverse=True)[:limit]
    return ";".join(f"{word}:{score:.5f}" for word, score in ordered)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "test_predictions.csv")
    if "gold_label" not in frame.columns and "label" in frame.columns:
        frame["gold_label"] = frame["label"]
    frame["id"] = frame["id"].astype(str)
    return frame


def identity_mask(frame: pd.DataFrame, identity_vocabulary: set[str]) -> pd.Series:
    if "has_identity_term" in frame.columns:
        return frame["has_identity_term"].astype(bool)
    return frame["text"].apply(
        lambda text: any(
            word_in_vocabulary(token, identity_vocabulary)
            for token in re.findall(r"[a-z0-9<>]+", str(text).lower())
        )
    )


def build_example_set(
    identity_vocabulary: set[str],
    max_common_fp: int | None,
    max_resolved_fp: int | None,
    max_correct_hateful: int | None,
) -> pd.DataFrame:
    predictions = {name: load_predictions(run_dir) for name, run_dir in MODELS.items()}
    baseline = predictions["baseline"]
    false_positive_sets = {}
    correct_hateful_sets = {}

    for name, frame in predictions.items():
        has_identity = identity_mask(frame, identity_vocabulary)
        false_positive_sets[name] = set(
            frame.loc[
                (frame["gold_label"] == "non_hateful")
                & (frame["pred_label"] == "hateful")
                & has_identity,
                "id",
            ]
        )
        correct_hateful_sets[name] = set(
            frame.loc[
                (frame["gold_label"] == "hateful")
                & (frame["pred_label"] == "hateful")
                & has_identity,
                "id",
            ]
        )

    common_fp_ids = sorted(set.intersection(*false_positive_sets.values()))
    resolved_ids = sorted(false_positive_sets["baseline"] - false_positive_sets["aggressive_reduction"])
    correct_hateful_ids = sorted(set.intersection(*correct_hateful_sets.values()))
    if max_common_fp is not None:
        common_fp_ids = common_fp_ids[:max_common_fp]
    if max_resolved_fp is not None:
        resolved_ids = resolved_ids[:max_resolved_fp]
    if max_correct_hateful is not None:
        correct_hateful_ids = correct_hateful_ids[:max_correct_hateful]

    subsets = []
    for ids, subset_name in [
        (common_fp_ids, "common_identity_fp_all_models"),
        (resolved_ids, "baseline_identity_fp_resolved_by_aggressive"),
        (correct_hateful_ids, "common_correct_identity_hateful"),
    ]:
        subset = baseline[baseline["id"].isin(ids)].copy()
        subset["analysis_subset"] = subset_name
        subsets.append(subset)
    return pd.concat(subsets, ignore_index=True).drop_duplicates(["id", "analysis_subset"])


def summarize_specificity(long_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    share_columns = ["identity_share", "context_share", "other_share"]
    summary = (
        long_frame.groupby(["analysis_subset", "model"], as_index=False)[share_columns]
        .mean()
    )
    summary.to_csv(output_dir / "attribution_specificity_summary.csv", index=False)

    wide = summary.pivot(index="analysis_subset", columns="model", values=share_columns)
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in share_columns:
        for model in ["balanced_guardrail", "aggressive_reduction"]:
            baseline_column = f"{metric}_baseline"
            model_column = f"{metric}_{model}"
            if baseline_column in wide.columns and model_column in wide.columns:
                wide[f"{metric}_{model}_delta_vs_baseline"] = wide[model_column] - wide[baseline_column]
    wide.to_csv(output_dir / "attribution_specificity_delta_vs_baseline.csv", index=False)

    example_wide = long_frame.pivot_table(
        index=["analysis_subset", "id", "text"],
        columns="model",
        values=["identity_share", "context_share"],
    ).reset_index()
    example_wide.columns = ["_".join(str(part) for part in column if str(part)) for column in example_wide.columns]
    example_wide["aggressive_identity_delta"] = (
        example_wide["identity_share_aggressive_reduction"] - example_wide["identity_share_baseline"]
    )
    example_wide["aggressive_context_delta"] = (
        example_wide["context_share_aggressive_reduction"] - example_wide["context_share_baseline"]
    )
    example_wide.to_csv(output_dir / "attribution_specificity_examples_wide.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/attribution_specificity")
    parser.add_argument("--candidate-file", default="data/contrastive/false_positives_for_manual_pairs.csv")
    parser.add_argument("--run-attribution", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--max-common-fp", type=int, default=10)
    parser.add_argument("--max-resolved-fp", type=int, default=None)
    parser.add_argument("--max-correct-hateful", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "attribution_specificity_long.csv"
    summary_path = output_dir / "attribution_specificity_summary.csv"

    if summary_path.exists() and not args.force and not args.run_attribution:
        print(f"Using existing specificity summary: {summary_path}")
        return
    if args.run_attribution and not args.force:
        raise ValueError("--run-attribution requires --force because it overwrites attribution CSVs.")

    identity_vocabulary = build_identity_vocabulary(Path(args.candidate_file))
    if args.run_attribution:
        examples = build_example_set(
            identity_vocabulary,
            args.max_common_fp,
            args.max_resolved_fp,
            args.max_correct_hateful,
        )
        examples.to_csv(output_dir / "attribution_specificity_examples.csv", index=False)
        device = choose_device(args.device)
        rows = []
        for model_name, run_dir in MODELS.items():
            model_dir = run_dir / "best_model"
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            model.to(device)
            model.eval()
            for _, row in examples.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "id": row["id"],
                        "analysis_subset": row["analysis_subset"],
                        "gold_label": row["gold_label"],
                        "baseline_pred_label": row["pred_label"],
                        "text": row["text"],
                        **attribution_specificity_for_text(
                            row["text"],
                            model,
                            tokenizer,
                            device,
                            identity_vocabulary,
                            n_steps=args.n_steps,
                        ),
                    }
                )
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(long_path, index=False)

    if not long_path.exists():
        raise FileNotFoundError(f"Missing cached specificity attribution: {long_path}")
    summary = summarize_specificity(pd.read_csv(long_path), output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
