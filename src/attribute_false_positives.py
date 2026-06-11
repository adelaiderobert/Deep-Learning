from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer


HATEFUL_LABEL_ID = 1


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_token(token: str) -> str:
    return token.replace("Ġ", "").strip().lower()


def token_starts_new_word(token: str, current_word: str) -> bool:
    return token.startswith("Ġ") or current_word == ""


def merge_tokens_to_words(scored_tokens: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    words = []
    current_word = ""
    current_score = 0.0

    for item in scored_tokens:
        raw_token = str(item["raw_token"])
        token_piece = clean_token(raw_token)
        score = float(item["score"])
        if not token_piece:
            continue

        if token_starts_new_word(raw_token, current_word):
            if current_word:
                words.append({"word": current_word, "score": current_score})
            current_word = token_piece
            current_score = score
        else:
            current_word += token_piece
            current_score += score

    if current_word:
        words.append({"word": current_word, "score": current_score})
    return words


def attribution_forward(model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


def attribute_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    n_steps: int,
) -> tuple[list[dict[str, float | str]], list[str]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_ids[:, 0] = input_ids[:, 0]
    if input_ids.shape[1] > 1:
        baseline_ids[:, -1] = input_ids[:, -1]

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids)
    baseline_embeds = embedding_layer(baseline_ids)

    ig = IntegratedGradients(lambda embeds, mask: attribution_forward(model, embeds, mask))
    attributions = ig.attribute(
        inputs=inputs_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask,),
        target=HATEFUL_LABEL_ID,
        n_steps=n_steps,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

    scored_tokens = []
    for token, score in zip(tokens, token_scores.tolist(), strict=True):
        if token in tokenizer.all_special_tokens:
            continue
        cleaned = clean_token(token)
        if cleaned:
            scored_tokens.append({"raw_token": token, "token": cleaned, "score": float(score)})
    return scored_tokens, tokens


def summarize_top_tokens(scored_tokens: list[dict[str, float | str]], top_k: int) -> list[dict[str, float | str]]:
    return sorted(scored_tokens, key=lambda item: abs(float(item["score"])), reverse=True)[:top_k]


def summarize_top_words(scored_words: list[dict[str, float | str]], top_k: int) -> list[dict[str, float | str]]:
    return sorted(scored_words, key=lambda item: abs(float(item["score"])), reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Baseline run directory, e.g. outputs/baseline/run3.")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "best_model"
    input_file = Path(args.input_file) if args.input_file else run_dir / "validation_identity_false_positives.csv"

    frame = pd.read_csv(input_file)
    if args.max_examples is not None:
        frame = frame.head(args.max_examples).copy()

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    rows = []
    token_counter = Counter()
    token_score_totals = defaultdict(float)
    word_counter = Counter()
    word_score_totals = defaultdict(float)

    for _, example in frame.iterrows():
        scored_tokens, _ = attribute_text(
            example["text"],
            model,
            tokenizer,
            device,
            args.max_length,
            args.n_steps,
        )
        top_tokens = summarize_top_tokens(scored_tokens, args.top_k)
        top_words = summarize_top_words(merge_tokens_to_words(scored_tokens), args.top_k)
        top_token = str(top_tokens[0]["token"]) if top_tokens else ""
        top_token_score = float(top_tokens[0]["score"]) if top_tokens else 0.0
        top_word = str(top_words[0]["word"]) if top_words else ""
        top_word_score = float(top_words[0]["score"]) if top_words else 0.0

        for item in top_tokens:
            token = str(item["token"])
            token_counter[token] += 1
            token_score_totals[token] += abs(float(item["score"]))
        for item in top_words:
            word = str(item["word"])
            word_counter[word] += 1
            word_score_totals[word] += abs(float(item["score"]))

        rows.append(
            {
                "id": example["id"],
                "text": example["text"],
                "gold_label": example.get("gold_label", example.get("label", "")),
                "pred_label": example.get("pred_label", ""),
                "prob_hateful": example.get("prob_hateful", ""),
                "identity_terms": example.get("identity_terms", ""),
                "top_token": top_token,
                "top_token_score": top_token_score,
                "top_tokens": ";".join(f"{item['token']}:{float(item['score']):.6f}" for item in top_tokens),
                "top_word": top_word,
                "top_word_score": top_word_score,
                "top_words": ";".join(f"{item['word']}:{float(item['score']):.6f}" for item in top_words),
            }
        )

    attribution_df = pd.DataFrame(rows)
    attribution_df.to_csv(run_dir / "false_positive_attributions.csv", index=False)

    token_rows = [
        {
            "token": token,
            "top_k_count": count,
            "mean_abs_score_when_top_k": token_score_totals[token] / count,
        }
        for token, count in token_counter.most_common()
    ]
    word_rows = [
        {
            "word": word,
            "top_k_count": count,
            "mean_abs_score_when_top_k": word_score_totals[word] / count,
        }
        for word, count in word_counter.most_common()
    ]
    token_df = pd.DataFrame(token_rows)
    word_df = pd.DataFrame(word_rows)
    token_df.to_csv(run_dir / "top_attributed_tokens.csv", index=False)
    word_df.to_csv(run_dir / "top_attributed_words.csv", index=False)

    print(f"Saved attributions for {len(attribution_df)} examples")
    print(f"Saved aggregated token table with {len(token_df)} tokens")
    print(f"Saved aggregated word table with {len(word_df)} words")


if __name__ == "__main__":
    main()
