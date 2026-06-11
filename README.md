# Deep-Learning
Contrastive fine-tuning pipeline for hate speech detection using RoBERTa. Uses Integrated Gradients to diagnose identity-related false positives, then applies a mixed contrastive-CE loss over targeted pairs. 208 hyperparameter configs evaluated on HateXplain. EE-559 Deep Learning, EPFL 2026.

# Model-Error-Driven Contrastive Fine-Tuning for Robust Hate Speech Detection
The project combines a written paper, a poster, and a fully reproducible Python pipeline.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                                        # Raw HateXplain JSON files (not included — see below)
│   ├── processed/                                  # Binary train/validation/test CSVs (regenerate with src/inspect_hatexplain.py)
│   └── contrastive/
│       ├── contrastive_pairs_completed.csv         # 128 final (non-hateful, hateful) pairs
│       └── false_positives_for_manual_pairs.csv    # 64 validation false positives used as sources
├── notebooks/
│   └── 01_pipeline_progress.ipynb                 # Main interactive narrative — start here
├── outputs/
│   ├── baseline/run1/                             # Baseline metrics, predictions, attribution files (no weights)
│   ├── contrastive/
│   │   ├── all_model_comparison_clean.csv         # Consolidated grid results (208 configs)
│   │   ├── balanced_for_attribution_m18_a0p2_b0p05/
│   │   └── aggressive_for_attribution_m22_a2_b0p5/
│   ├── attribution_comparison/
│   ├── attribution_specificity/
│   ├── false_negative_analysis/
│   └── plots/                                     # Final figures for paper and poster
├── src/
│   ├── inspect_hatexplain.py                      # Dataset preprocessing and identity vocabulary
│   ├── train_baseline.py                          # Baseline RoBERTa training
│   ├── extract_false_positives.py                 # Inference and false-positive extraction
│   ├── attribute_false_positives.py               # Integrated Gradients attribution
│   ├── train_contrastive.py                       # Mixed-loss contrastive fine-tuning
│   ├── analyze_grid_search.py                     # Consolidate grid-search results
│   ├── analyze_attribution_shift.py               # Attribution comparison across models
│   ├── analyze_false_negatives.py                 # False-negative guardrail analysis
│   ├── analyze_attribution_specificity.py         # Identity vs context attribution breakdown
│   └── plot_final_analysis.py                     # Generate final figures
├── paper.pdf                                      # Written report
├── poster.pdf                                     # Project poster
├── Screencast.mp4                                 # Demo screencast
├── requirements.txt
└── README.md
```

---

## Project — Contrastive Fine-Tuning for Identity Bias Mitigation

**Research question:** Can model-error-driven contrastive fine-tuning reduce false positives on identity-related non-hateful text without substantially degrading recall on genuinely hateful content?

Automated hate-speech classifiers often exploit surface-level lexical cues rather than genuine hateful intent. Identity terms such as *Muslim*, *Black*, or *gay* become spurious shortcuts during training, causing non-hateful statements mentioning these groups to be misclassified. Generic counterfactual data augmentation has been shown to backfire when pairs are not grounded in the model's actual errors. Our pipeline starts from the classifier's own mistakes and builds targeted pairs to correct them.

**Dataset:** HateXplain ([Mathew et al., 2021](https://arxiv.org/abs/2012.10289)) — binary labels derived by majority vote over per-annotator annotations, with offensive-language posts excluded. Final splits: 10,999 / 1,374 / 1,376 (train / validation / test).

> Raw JSON files are not included. Place `hatexplain_dataset.json` and `hatexplain_post_id_divisions.json` under `data/raw/` before running the pipeline (download instructions below).

**Pipeline — four stages:**

1. **Baseline** — fine-tune `roberta-base` on HateXplain for binary hate/non-hate classification (3 epochs, batch size 16, lr 2×10⁻⁵). Reaches test Macro-F1 0.895 with 70 false positives and 26 identity false positives.

2. **Error diagnosis** — run Integrated Gradients on the hateful logit of all 71 validation false positives. Attribution scores surface identity trigger terms and slur variants beyond the initial seed vocabulary, yielding a final vocabulary of 53 terms matching 64 candidates.

3. **Contrastive pair construction** — each of the 64 candidates is rewritten twice into a (non-hateful source, hateful counterpart) pair. Both examples preserve the identity term and sentence length; only the semantic intent changes. LLM-generated counterparts were manually reviewed. Final dataset: 128 pairs.

4. **Contrastive fine-tuning** — second-stage training from the baseline checkpoint with a mixed loss:

$$\mathcal{L} = \mathcal{L}^{\text{train}}_{\text{CE}} + \beta\,\mathcal{L}^{\text{pairs}}_{\text{CE}} + \alpha\,\mathcal{L}_{\text{con}}$$

where the contrastive term pushes paired embeddings apart by at least margin $m$:

$$\mathcal{L}_{\text{con}} = \max\!\left(0,\; m - \|z_{\text{non-hateful}} - z_{\text{hateful}}\|_2\right)^2$$

Hyperparameters $\alpha$, $\beta$, and $m$ are swept over 208 configurations on the EPFL RCP cluster.

---

## Key Results

| Model | ID-FP ↓ | FP | FN | Recall ↑ | Macro-F1 ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 26 | 70 | 72 | 0.879 | 0.895 |
| Balanced (m=18, α=0.2, β=0.05) | 22 | 63 | 83 | 0.860 | 0.891 |
| Aggressive (m=22, α=2.0, β=0.5) | 17 | 47 | 103 | 0.827 | 0.887 |

ID-FP = identity false positives (non-hateful identity-related examples misclassified as hateful).

The method provides a controllable false-positive / false-negative trade-off. Stronger contrastive pressure ($m \uparrow$, $\alpha \uparrow$) reduces identity false positives but makes the classifier more conservative, increasing false negatives. The balanced setting reduces ID-FP by 15% with a 1.9-point recall drop; the aggressive setting achieves a 35% reduction at the cost of a 5.2-point recall drop. Attribution analysis confirms that reliance on identity terms decreases, though attribution mass shifts broadly rather than concentrating on hateful-context words.

---

## Dependencies

```
torch
transformers
captum
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

Requires **Python 3.9+**.

---

## Running the Pipeline

All precomputed result files (metrics, predictions, attribution CSVs, plots) are stored in `outputs/` and loaded directly by the notebook — no retraining is needed to browse the results. Model weights are not included due to file size; retrain from scratch using the steps below.

```bash
# Download raw data
mkdir -p data/raw
wget -O data/raw/hatexplain_dataset.json \
    https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
wget -O data/raw/hatexplain_post_id_divisions.json \
    https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json

# Step 1 — Preprocess
python src/inspect_hatexplain.py

# Step 2 — Train baseline
python src/train_baseline.py --epochs 3 --batch-size 16

# Step 3 — Extract false positives
python src/extract_false_positives.py --run-dir outputs/baseline/run1

# Step 4 — Integrated Gradients attribution
python src/attribute_false_positives.py \
    --run-dir outputs/baseline/run1 \
    --input-file outputs/baseline/run1/validation_false_positives.csv

# Step 5 — Contrastive fine-tuning (balanced config)
python src/train_contrastive.py \
    --init-model-dir outputs/baseline/run1/best_model \
    --pair-file data/contrastive/contrastive_pairs_completed.csv \
    --margin 18 --alpha 0.2 --beta 0.05

# Step 6 — Analysis and figures
python src/analyze_grid_search.py --force
python src/analyze_attribution_shift.py --force
python src/analyze_false_negatives.py --force
python src/analyze_attribution_specificity.py --force
python src/plot_final_analysis.py --force
```

The full 208-configuration grid search was run on the EPFL RCP cluster. Step 5 shows a single example run; reproducing the full grid requires cluster access (see notebook Section 10).

---

## Usage of AI Tools

As required by the course guidelines, any use of AI assistants (ChatGPT, Claude, etc.) is documented directly in the paper and in the relevant code comments.

---

## License

Academic work — EPFL EE-559, 2026. Not intended for redistribution.
