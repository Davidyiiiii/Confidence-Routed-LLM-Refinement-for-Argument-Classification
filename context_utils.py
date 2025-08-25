# context_utils.py
import nltk
from bisect import bisect_right
from nltk import data
from typing import List, Tuple, Dict
# 自动检查 punkt & punkt_tab
for res in ("punkt", "punkt_tab"):
    try:
        data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res, quiet=True)# --- 句子切分并记录字符区间 -------------------------------




def get_sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    spans, offset = [], 0
    for sent in nltk.sent_tokenize(text):
        start = text.find(sent, offset)
        end   = start + len(sent)
        spans.append((start, end, sent))
        offset = end
    return spans


def extract_context(full_text: str,
                    span_start: int,
                    span_end: int,
                    k: int = 2,
                    win: int = 150) -> Dict[str, str]:
    """
    先尝试句子级 (±k 句)，失败则退回字符窗口 (±win 字符)
    """
    sents = get_sentence_spans(full_text)

    try:
        idx = next(i for i, (s, e, _) in enumerate(sents) if s <= span_start < e)

        span_sent  = sents[idx][2]
        prev_sents = " ".join(s[2] for s in sents[max(0, idx-k): idx])
        next_sents = " ".join(s[2] for s in sents[idx+1 : idx+1+k])

        # 若 prev/next 全空，再用字符窗口补一遍
        if not prev_sents and not next_sents:
            raise ValueError
        return {"span": span_sent, "prev": prev_sents, "next": next_sents}

    except (StopIteration, ValueError):
        # —— fallback: raw character window ——
        prev_raw = full_text[max(0, span_start - win): span_start].strip()
        next_raw = full_text[span_end: span_end + win].strip()
        span_txt = full_text[span_start: span_end]

        return {"span": span_txt, "prev": prev_raw, "next": next_raw}