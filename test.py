"""
predict_test_full.py
────────────────────────────────────────────────────────
load
  • persuade_arrow_full/      ← 01_build_arrow_full.py
  • bert_full512/best/        ← best checkpoint
test split →  CSV
"""

from pathlib import Path
import numpy as np, pandas as pd, torch
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer)

# ---------- path ----------
DATA_DIR  = "persuade_arrow_full"                  # arrow
CKPT_DIR  = Path("bert_full/best")              # best weight
RAW_TEST  = Path("dataset/persuade_corpus_2.0_test.csv")
OUT_CSV   = Path("bert_full/test_pred_full.csv")
# ------------------------

# load Fine-tuned Bert model
ds = load_from_disk(DATA_DIR)
test_ds = ds["test"]

tok = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
trainer = Trainer(model, tokenizer=tok)

# prediction
print(" Predicting…")
logits = trainer.predict(test_ds).predictions
conf   = torch.softmax(torch.tensor(logits), -1).max(-1).values.numpy()
pred   = np.argmax(logits, axis=-1)
gold   = test_ds["label"]

#  essay_id / discourse_text
raw = pd.read_csv(RAW_TEST, low_memory=False)
raw = raw[raw.discourse_type != "Unannotated"].reset_index(drop=True)

assert len(raw) == len(pred), "ERRO"

# Save
df_out = pd.DataFrame({
    "essay_id":        raw["essay_id"],
    "discourse_id":   raw["discourse_id"],
    "discourse_text":  raw["discourse_text"],
    "gold":            gold,
    "pred":            pred,
    "conf":            conf,
    "discourse_start": raw["discourse_start"],
    "discourse_end":   raw["discourse_end"],
    "full_text":       raw["full_text"]
})

df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
print("Save in", OUT_CSV)

