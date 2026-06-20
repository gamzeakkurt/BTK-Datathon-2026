# 🏆 Datathon 2026 8th Place Solution Writeup — Career Success Score Prediction 

> **Final Private Leaderboard:** **81.849**  
> **Public Leaderboard:** **81.561**

Our solution for the **Datathon 2026 Career Success Score Prediction** competition.

The objective was to predict a student's **career success score** (0–100) using structured demographic and educational features together with Turkish mentor feedback text.

---

## Overview

The dataset consists of:

- **10,000 students**
- **46 structured tabular features**
- **1 Turkish free-text feature** (`mentor_feedback_text`)
- Target:
  - `career_success_score` (continuous, 0–100)

Unlike a standard regression problem, evaluation used a **year-weighted Mean Squared Error**, where each example is weighted according to the ratio between its application year's frequency in the test and training distributions.

This introduces a **covariate shift toward recent application years**, making generalization more important than simply minimizing validation error.

---

## Evaluation Metric

Rows are weighted by

\[
W_{year}=\frac{f_{test}^{year}}{f_{train}^{year}}
\]

and normalized so that

\[
\bar{W}=1
\]

The competition therefore rewards models that remain robust under changing year distributions rather than models that overfit historical data.

---

# Final Solution

Our final submission is a convex ensemble of three complementary models.

| Weight | Model | Description |
|---------|-------|-------------|
| **0.512** | **p2qwfp** | 3-seed RealTabPFN with leak-free pseudo-labeling and Qwen text embeddings |
| **0.426** | **ag_emb** | TabPFN embeddings (512 dimensions) → AutoGluon (medium preset, year-weighted) |
| **0.062** | **ftext** | Hand-crafted formula baseline + LightGBM residual model trained on sparse text |

Final predictions are clipped to the valid score range:

```text
prediction = clip(weighted_ensemble, 0, 100)
```

---

# Results

| Leaderboard | Score |
|-------------|------:|
| Public | **81.561** |
| Private | **81.849** |

This was our best-performing submission on the final leaderboard.

---

# Model Components

## 1. RealTabPFN + Qwen Embeddings

Our strongest individual model.

Features include:

- Tabular features
- Qwen embeddings extracted from Turkish mentor feedback
- Leak-free pseudo-labeling
- Three independent random seeds
- Prediction averaging

This model contributes just over half of the final ensemble weight.

---

## 2. TabPFN Embeddings → AutoGluon

Instead of using TabPFN directly for prediction, we use it as a feature extractor.

Pipeline:

```
Tabular Features
        ↓
   TabPFN Encoder
        ↓
 512-dimensional embeddings
        ↓
   AutoGluon Regressor
        ↓
 Weighted prediction
```

Training uses the competition's year-weighted objective to better match the evaluation metric.

---

## 3. Formula + Sparse Text Residual

A lightweight complementary model consisting of:

- a hand-engineered baseline formula
- sparse text features
- LightGBM trained only on the remaining residual

Although this model is individually weaker, it provides useful diversity within the ensemble.

---

# Ensemble

Final prediction:

```text
0.512 × p2qwfp
+ 0.426 × ag_emb
+ 0.062 × ftext
```

followed by

```text
clip(prediction, 0, 100)
```

The models make different types of errors, allowing the ensemble to consistently outperform any individual model.

---

# Key Takeaway

The biggest lesson from this competition was **not** discovering a more complex model.

Once performance approached the dataset's noise floor, **robustness mattered more than squeezing out tiny validation gains**.

Our simplest embedding-based blend (internally referred to as **v15**) exhibited:

- the smallest public → private leaderboard degradation,
- the strongest generalization under year distribution shift,
- and ultimately outperformed more aggressively tuned variants on the private leaderboard.

Sometimes, the least-overfit ensemble wins.

---



# Reproducing the Solution

1. Prepare the competition dataset.
2. Generate Qwen embeddings for `mentor_feedback_text`.
3. Train the RealTabPFN models.
4. Generate TabPFN embeddings and train AutoGluon.
5. Train the sparse-text residual model.
6. Blend predictions using the published weights.

---

# Acknowledgements

Thanks to the BTK AKADEMİ organizers for creating an interesting regression task combining structured data, natural language, and distribution shift.
