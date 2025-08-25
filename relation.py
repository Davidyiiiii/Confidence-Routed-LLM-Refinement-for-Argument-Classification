import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# ──  CSV path ───────────────────────────────
CSV_PATH = "bert_full/final1.csv"
# CSV_PATH = "bert_full/test_pred_hybrid.csv"
# ────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

df["correct"] = df["deepseek"] == df["gold"]

# plt.figure(figsize=(6,4))
# plt.hist(df.loc[df["correct"], "conf"], bins=20, alpha=0.6, label="correct")
# plt.hist(df.loc[~df["correct"], "conf"], bins=20, alpha=0.6, label="wrong")
# plt.xlabel("confidence")
# plt.ylabel("count")
# plt.title("Confidence distribution")
# plt.legend()
# plt.tight_layout()
# plt.show()

bins = pd.cut(df["conf"], bins=[0,0.2,0.4,0.6,0.8,1.0], right=False)
acc_by_bin = df.groupby(bins)["correct"].mean()

plt.figure(figsize=(6,4))
acc_by_bin.plot(kind="bar", color="steelblue")
plt.ylim(0,1)
plt.ylabel("accuracy")
plt.title("Accuracy by confidence")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, f1_score



from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

gold, pred = df.gold.values, df.deepseek.values


acc = accuracy_score(gold, pred)
f1m = f1_score(gold, pred, average="macro")
per_f1 = f1_score(gold, pred, average=None)

print("Overall Acc :", f"{acc:.4f}")
print("Macro-F1    :", f"{f1m:.4f}\n")
print(classification_report(
    gold, pred,
    target_names=["Lead","Pos","Claim","Ctr","Reb","Evid","Conc"],
    digits=4)
)


cm = confusion_matrix(gold, pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(range(7), ["Lead","Pos","Claim","Ctr","Reb","Evid","Conc"],
           rotation=45, ha="right", fontsize=8)
plt.yticks(range(7), ["Lead","Pos","Claim","Ctr","Reb","Evid","Conc"],
           fontsize=8)
for i in range(7):
    for j in range(7):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color="white" if cm[i,j] > cm.max()*0.6 else "black",
                 fontsize=7)
plt.tight_layout()
plt.show()

# ③  F1 for each class
labels = ["Lead","Position","Claim","Counter","Rebuttal","Evidence","Conclude"]
plt.figure(figsize=(7,4))
plt.bar(labels, per_f1, width=0.6)
plt.ylim(0,1); plt.ylabel("F1")
plt.title("Per-Class F1")
for i,v in enumerate(per_f1):
    plt.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.show()