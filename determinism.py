"""
Determinism guarantees for reproducible evaluation runs.

Every evaluation run must produce identical scores given identical inputs.
This module centralises all seeding and hashing so determinism is
enforced in one place rather than scattered across the codebase.

Strategies used
---------------
1. Global RNG seeding (Python random, NumPy, PyTorch).
2. Fixed Whisper decode options (temperature=0, greedy beam search).
3. Fixed Ollama options (temperature=0, seed=N).
4. SHA-256 content hashing of all inputs + config for audit trail.
5. Canonical text normalisation applied before every metric computation.
"""

import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

_SEEDED = False


def seed_everything(seed: int = 42) -> None:
    """
    Seed all RNG sources for full reproducibility.
    Safe to call multiple times; re-seeds every time.
    """
    global _SEEDED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (optional — may not be installed in all environments)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("PyTorch not available — skipping torch seeding")

    _SEEDED = True
    logger.debug(f"All RNG sources seeded with seed={seed}")


def assert_seeded() -> None:
    """Raise RuntimeError if seed_everything() has not been called."""
    if not _SEEDED:
        raise RuntimeError(
            "Determinism not initialised. Call seed_everything(seed) "
            "before running any evaluation."
        )


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def hash_text(text: str) -> str:
    """Return SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file's binary content."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    """
    Return a stable SHA-256 hash of a config dict.
    Keys are sorted so insertion order doesn't affect the hash.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_test_case(test_case: dict[str, Any]) -> str:
    """Hash the immutable fields of a test case for audit logging."""
    fields = {
        "id": test_case.get("id", ""),
        "input_text": test_case.get("input_text", ""),
        "expected_response": test_case.get("expected_response", ""),
        "reference_facts": test_case.get("reference_facts", []),
    }
    return hash_config(fields)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Contraction expansion table (US English)
_CONTRACTIONS: dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "it'd": "it would",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
    re.IGNORECASE,
)


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    expand_contractions: bool = True,
    strip_extra_whitespace: bool = True,
) -> str:
    """
    Apply a canonical normalisation pipeline to text before metric scoring.

    The same normalisation is applied to both hypothesis and reference,
    so minor formatting differences don't penalise the score.

    Steps (in order):
    1. Unicode NFKC normalisation
    2. Contraction expansion
    3. Lowercase
    4. Punctuation removal
    5. Whitespace collapse
    """
    if not text:
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Contraction expansion
    if expand_contractions:
        text = _CONTRACTION_RE.sub(
            lambda m: _CONTRACTIONS[m.group(0).lower()], text
        )

    # 3. Lowercase
    if lowercase:
        text = text.lower()

    # 4. Remove punctuation (keep spaces and alphanumeric)
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)

    # 5. Whitespace collapse
    if strip_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def get_fixed_whisper_options() -> dict[str, Any]:
    """
    Return Whisper decode options that guarantee deterministic transcription.
    temperature=0 forces greedy decoding; beam_size/best_of are fixed.
    """
    return {
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,  # Prevents compounding drift
        "fp16": False,
    }


def get_fixed_ollama_options(seed: int = 42) -> dict[str, Any]:
    """
    Return Ollama inference options that guarantee deterministic outputs.
    temperature=0 + top_k=1 = greedy sampling = deterministic.
    """
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": seed,
        "num_predict": 512,
    }
