
from pathlib import Path
import os, time, json, requests
import pandas as pd
from tqdm import tqdm
import openai
from sklearn.metrics import accuracy_score, f1_score

# ---------- Path ----------
CSV_IN = Path("bert_full/marked.csv")  # ←  marked text
CSV_OUT = Path("bert_full/Refinement.csv")
THRESH_GPT = 0.50                                      # conf < 0.50 → GPT
THRESH_DEEPSEEK = 0.80                                 # 0.50 ≤ conf < 0.80 → DeepSeek
SLEEP    = 0.6

N_PER_BIN = int(os.getenv("N_PER_BIN",3 ))
SLEEP    = float(os.getenv("SLEEP", "0.6"))

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-o3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DeepSeek
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

SYSTEM_MSG = """
You are an expert discourse-analysis assistant.

LABEL SET
0 Lead                — Introduces the topic or hooks the reader
1 Position            — States the author’s overall stance
2 Claim               — A reason that supports the position
3 Counterclaim        — A reason that opposes the position
4 Rebuttal            — Refutes or weakens a counterclaim
5 Evidence            — Fact, statistic, quote, or example supporting a claim
6 Concluding Statement— Wraps up or summarizes the argument

TASK
----
The user will send the FULL essay text, but with exactly ONE focus span
surrounded by >>> <<<.
Your job is to assign a label **only for that focus span**.

OUTPUT FORMAT
• Reply with the single digit 0-6 on the FIRST line.
• No additional text, no words, no punctuation.
• If unsure, choose the SINGLE most plausible label.
""".strip()

def to_digit(val):
    if isinstance(val, int) and 0 <= val <= 6:
        return val
    if isinstance(val, str):
        tok = val.strip().split()[0]
        if tok.isdigit():
            d = int(tok)
            return d if 0 <= d <= 6 else None
    return None

_openai_client = None
def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai  # type: ignore
        if not OPENAI_API_KEY:
            raise EnvironmentError("Missing OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY
        _openai_client = openai
    return _openai_client

def ask_gpt(marked_text: str):
    try:
        openai = _get_openai()
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": f'Essay:\\n"""\\n{marked_text}\\n"""'}
            ]
        )
        txt = resp.choices[0].message.content
        return to_digit(txt)
    except Exception as e:
        tqdm.write(f"[GPT ERROR] {e}")
        return None

def ask_deepseek(marked_text: str):
    if not DEEPSEEK_API_KEY:
        tqdm.write("Missing DEEPSEEK_API_KEY, skip DeepSeek refinement")
        return None
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": f'Essay:\\n"""\\n{marked_text}\\n"""'},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }
    try:
        r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        return to_digit(txt)
    except Exception as e:
        tqdm.write(f"[DeepSeek ERROR] {e}")
        return None

def main():
    df = pd.read_csv(CSV_IN)
    required = {"gold", "pred", "conf", "marked_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing：{sorted(missing)}")

    # 初始化列
    df["final"] = df["pred"]
    if "gpt" not in df.columns: df["gpt"] = df["pred"]
    if "deepseek" not in df.columns: df["deepseek"] = df["pred"]

 
    mask_gpt = (df["conf"] < THRESH_GPT)
    mask_ds  = (df["conf"] >= THRESH_GPT) & (df["conf"] < THRESH_DEEPSEEK)

    idxs_gpt = list(df.index[mask_gpt])
    idxs_ds  = list(df.index[mask_ds])

    tqdm.write(f"OpenAI o3  (conf < {THRESH_GPT:.2f})：Total Number:{mask_gpt.sum()} .")
    for idx in tqdm(idxs_gpt, desc="Open AI o3 refine", total=len(idxs_gpt)):
        row = df.loc[idx]
        label = ask_gpt(row.marked_text)
        if label is not None:
            df.at[idx, "gpt"] = label
            df.at[idx, "final"] = label
            tqdm.write(f"[{idx}] True={row.gold} Bert={row.pred} Confidence={row.conf:.3f} OpenAI={label} ")
        else:
            tqdm.write(f"[{idx}] GPT invalid outputs，Keep Bert pred={row.pred}")
        time.sleep(SLEEP)

    tqdm.write(f"DeepSeek V3 ({THRESH_GPT:.2f} ≤ conf < {THRESH_DEEPSEEK:.2f})：Total Number: {mask_ds.sum()}")
    for idx in tqdm(idxs_ds, desc="DeepSeek V3 refine", total=len(idxs_ds)):
        row = df.loc[idx]
        label = ask_deepseek(row.marked_text)
        if label is not None:
            df.at[idx, "deepseek"] = label
            df.at[idx, "final"] = label
            tqdm.write(f"[{idx}] True={row.gold} Bert={row.pred} Confidence={row.conf:.3f} Deepseek={label} ")
        else:
            tqdm.write(f"[{idx}] DeepSeek invalid outputs，Ke pred={row.pred}")
        time.sleep(SLEEP)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    tqdm.write(f"Save → {CSV_OUT}")
if __name__ == "__main__":
    main()