import math
import re
from collections import Counter
from typing import Dict, List, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")] 


def _vectorize(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    if not vec_a or not vec_b:
        return 0.0
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity(resume_text: str, job_text: str) -> float:
    tokens_a = _tokenize(resume_text)
    tokens_b = _tokenize(job_text)
    vec_a = _vectorize(tokens_a)
    vec_b = _vectorize(tokens_b)
    return _cosine_similarity(vec_a, vec_b)

