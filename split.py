# export_split_to_csv.py
from pathlib import Path
from datasets import load_from_disk

ARROW_DIR = "persuade_arrow_full"
OUT_DIR   = Path("arrow_csv"); OUT_DIR.mkdir(exist_ok=True)

ds = load_from_disk(ARROW_DIR)               # DatasetDict(train/validation/test)

for split in ("train", "validation", "test"):
    df = ds[split].to_pandas()               # Arrow → pandas
    out_path = OUT_DIR / f"{split}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✓ {split:10s} → {out_path}  ({len(df):,} rows)")
