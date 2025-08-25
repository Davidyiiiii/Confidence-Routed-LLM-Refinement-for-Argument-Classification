"""
train_bert_full.py
────────────────────────────────────────────────────────
• load persuade_arrow_full/ data（full essay+ >>>discourse<<<）
• fine-tune BERT-base（512 token）
•  val-macro-F1，save checkpoint
• Generate loss / val-metrics / confusion-matrix / per-class-F1 graph
"""

from pathlib import Path
import json, numpy as np, torch, matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)

# ─────────── Path and Hyper Parameter───────────
DATA_DIR = "persuade_arrow_full"      # ← 01_build_arrow_full.py
OUT_DIR  = Path("bert_full")
FIG_DIR  = OUT_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL    = "bert-base-uncased"
MAX_LEN  = 512
BATCH    = 8
EPOCHS   = 6
LR       = 3e-5
SEED     = 42
# ────────────────────────────────

#
ds = load_from_disk(DATA_DIR)
print("Train / Val / Test size:", len(ds["train"]), len(ds["validation"]), len(ds["test"]))

# Model & tokenizer
cfg   = AutoConfig.from_pretrained(MODEL, num_labels=7)
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=cfg)

if torch.cuda.is_available():
    model.to("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

def metric_fn(eval_pred):
    lg, lb = eval_pred
    pr = np.argmax(lg, -1)
    return {
        "accuracy": accuracy_score(lb, pr),
        "macro_f1": f1_score(lb, pr, average="macro")
    }

# load parameter
args = TrainingArguments(
    output_dir=OUT_DIR.as_posix(),
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=max(2, BATCH//2),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    eval_accumulation_steps=32,
    seed=SEED,
    logging_dir=(OUT_DIR/"logs").as_posix(),
    logging_strategy="steps",
    logging_steps=200,
    dataloader_num_workers=4,   # Windows 稳定设置
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=metric_fn,
)

# Training
from multiprocessing import freeze_support
if __name__ == "__main__":
    freeze_support()          # for Windows
    trainer.train()

    # ---- Save----
    best_dir = OUT_DIR / "best"
    trainer.save_model(best_dir.as_posix())
    print("Best Weight", best_dir)

    val_logits = trainer.predict(ds["validation"]).predictions
    val_conf   = torch.softmax(torch.tensor(val_logits), -1).max(-1).values.numpy()
    gamma = float(np.percentile(val_conf, 20))
    with open(OUT_DIR/"gamma.json", "w") as f:
        json.dump({"percentile": 20, "gamma": gamma}, f, indent=2)
    print(f"γ (20 %) = {gamma:.4f}")

    # Training Graph
    hist = trainer.state.log_history
    steps, tr_loss = zip(*[(h["step"], h["loss"]) for h in hist if "loss" in h])
    ep, v_acc, v_f1 = zip(*[(h["epoch"], h["eval_accuracy"], h["eval_macro_f1"])
                            for h in hist if "eval_accuracy" in h])
    plt.figure(); plt.plot(steps, tr_loss)
    plt.title("Training Loss"); plt.xlabel("Step"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(FIG_DIR/"loss_curve.png"); plt.close()

    plt.figure(); plt.plot(ep, v_acc, label="Val-Acc")
    plt.plot(ep, v_f1, label="Val-F1")
    plt.legend(); plt.title("Validation Metrics")
    plt.xlabel("Epoch"); plt.tight_layout()
    plt.savefig(FIG_DIR/"val_metrics.png"); plt.close()

    # 6️⃣  Confusion Matrices & per-class F1
    test_logits = trainer.predict(ds["test"]).predictions
    pred = np.argmax(test_logits, -1)
    gold = ds["test"]["label"]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gold, pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.xticks(range(7)); plt.yticks(range(7))
    plt.tight_layout(); plt.savefig(FIG_DIR/"confusion_matrix.png"); plt.close()

    f1_per = f1_score(gold, pred, average=None)
    cls = ["Lead","Pos","Claim","Ctr","Rebut","Evid","Conc"]
    plt.figure(figsize=(7,4))
    plt.bar(cls, f1_per); plt.ylim(0,1)
    plt.title("Per-Class F1"); plt.tight_layout()
    plt.savefig(FIG_DIR/"per_class_f1.png"); plt.close()

    print("All graph", FIG_DIR)
