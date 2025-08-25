Argument  Classification with Confidence-Routed LLM Refinement

This report trains a BERT classifier for argumentative discourse roles and routes low-confidence test predictions to LLMs (OpenAI / DeepSeek) for second-pass “refinement.” It also includes utilities to build Arrow datasets, mark focus spans inside full essays, and visualize results.

Seven labels are used throughout:

0 = Lead · 1 = Position · 2 = Claim · 3 = Counterclaim · 4 = Rebuttal · 5 = Evidence · 6 = Concluding Statement

Folder & scripts at a glance

build_arrow_full.py — Build HuggingFace Arrow datasets from raw CSV.

split.py — Export Arrow splits to CSV (optional).

train.py — Fine-tune BERT on the training split.

test.py — Run inference on test split, produce pred and conf.

mark.py — Embed the target span in the full essay as >>> … <<< → marked.csv.

refinement.py — Confidence-based LLM cascade (OpenAI / DeepSeek) to revise low-confidence rows.

relation.py — Analysis & plots (accuracy by confidence, confusion matrix, per-class F1, etc.)
