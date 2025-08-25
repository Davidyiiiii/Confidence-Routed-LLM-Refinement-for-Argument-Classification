
from pathlib import Path
import argparse, re, pandas as pd

def clean_mark(text: str) -> str:
    """remove >>> <<< avoid repeat"""
    return re.sub(r">>>|<<<", "", text)

def mark_span(text: str, start: int, end: int) -> str:
    txt = clean_mark(text)
    return txt[:start] + ">>>" + txt[start:end] + "<<<" + txt[end:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  default="bert_full/test_pred_with_se.csv", dest="inp",  help="bert_full/test_pred_with_se.csv")
    ap.add_argument("--out", default="bert_full/marked.csv", dest="outp", help="bert_full/marked.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, low_memory=False)
    need_cols = {"full_text", "discourse_start", "discourse_end"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"CSV must have : {need_cols}")

    df["marked_text"] = [
        mark_span(str(ft), int(s), int(e))
        for ft, s, e in zip(df.full_text, df.discourse_start, df.discourse_end)
    ]

    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outp, index=False, encoding="utf-8")
    print(f"Marked Essays  → {args.outp}")

if __name__ == "__main__":
    main()
