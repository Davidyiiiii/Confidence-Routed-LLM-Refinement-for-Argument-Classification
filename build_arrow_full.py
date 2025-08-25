# 01_build_arrow_full.py
from pathlib import Path
import re, unicodedata as ud, pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path("dataset")
OUT_DIR  = Path("persuade_arrow_full")
TOKENIZER = "bert-base-uncased"
MAX_LEN   = 512

labels = ["Lead","Position","Claim","Counterclaim",
          "Rebuttal","Evidence","Concluding Statement"]
lab2id = {l:i for i,l in enumerate(labels)}

def clean(x:str)->str:
    return re.sub(r"\s+"," ",ud.normalize("NFKC",str(x))).strip()

def read(name):
    df = pd.read_csv(DATA_DIR/f"persuade_corpus_2.0_{name}.csv", low_memory=False)
    df = df[df.discourse_type!="Unannotated"].reset_index(drop=True)
    df["label"] = df.discourse_type.map(lab2id)
    return df

df_train, df_test = read("train"), read("test")

def mark(row):
    start, end = int(row.discourse_start), int(row.discourse_end)
    ft = row.full_text
    return ft[:start]+" >>>"+ft[start:end]+"<<< "+ft[end:]

for df in (df_train, df_test):
    df["input_text"] = df.apply(mark, axis=1)

tok = AutoTokenizer.from_pretrained(TOKENIZER)

def tok_fn(batch):
    return tok(batch["input_text"],
               truncation=True, padding="max_length", max_length=MAX_LEN)

def to_ds(df):
    ds = Dataset.from_pandas(df[["essay_id","input_text","label"]],
                             preserve_index=False)
    return ds.map(tok_fn, batched=True,
                  remove_columns=["essay_id","input_text"])

train_val = df_train.sample(frac=1, random_state=42)      # shuffle
cut = int(len(train_val)*0.9)
train_ds, val_ds = to_ds(train_val[:cut]), to_ds(train_val[cut:])
test_ds  = to_ds(df_test)

DatasetDict(train=train_ds, validation=val_ds, test=test_ds).save_to_disk(OUT_DIR)
print("Save", OUT_DIR)
