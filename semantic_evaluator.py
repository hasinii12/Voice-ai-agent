"""
Semantic Similarity Evaluator.

Uses a dual-mode approach for robust semantic scoring:

1. Embedding cosine similarity (sentence-transformers)
   - Encodes both hypothesis and reference with a fixed model
   - Computes cosine similarity between embedding vectors
   - Fast, deterministic, no external API needed

2. LLM judge (Ollama)
   - Asks the LLM to rate semantic equivalence on a 0-10 scale
   - Captures meaning beyond lexical similarity
   - Requires Ollama running locally

Final score = embedding_weight * cos_sim + llm_judge_weight * llm_score

Determinism guarantees:
- Embeddings: same model + same input → same vector (model weights frozen)
- LLM judge: temperature=0, top_k=1, seed=42 → deterministic output
"""

from __future__ import annotations

import re
from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvalResult
from utils.determinism import normalize_text, get_fixed_ollama_options
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt template for LLM judge
# ---------------------------------------------------------------------------

_SEMANTIC_JUDGE_PROMPT = """\
You are an expert evaluator of AI assistant responses.

Your task: Rate the semantic equivalence between two texts on a scale of 0 to 10.

Definition of semantic equivalence:
- 10: Identical meaning, all key information present
- 8-9: Same meaning, minor phrasing differences
- 6-7: Mostly equivalent, small factual gaps
- 4-5: Partially equivalent, some key info missing or different
- 2-3: Loosely related but substantively different
- 0-1: Unrelated or contradictory

REFERENCE (expected answer):
{reference}

HYPOTHESIS (AI response):
{hypothesis}

Respond with ONLY a single integer from 0 to 10. No explanation. No other text.
Score:"""


class SemanticEvaluator(BaseEvaluator):
    """
    Semantic similarity evaluator combining embedding cosine distance
    and LLM-based semantic judgement.

    Config keys
    -----------
    embedding_model   : str  (default "all-MiniLM-L6-v2")
    llm_judge         : bool (default True)
    llm_judge_weight  : float in [0,1] (default 0.4)
    embedding_weight  : float in [0,1] (default 0.6)
    ollama.host       : str  (default "http://localhost:11434")
    ollama.model      : str  (default "llama3")
    ollama.options    : dict (merged with fixed deterministic options)
    """

    name = "semantic_similarity"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._embedding_model_name = self.config.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )
        self._llm_judge = self.config.get("llm_judge", True)
        self._llm_weight = float(self.config.get("llm_judge_weight", 0.4))
        self._emb_weight = float(self.config.get("embedding_weight", 0.6))

        # Lazy-loaded to avoid slow import at startup
        self._embedding_model = None

        # Ollama config
        ollama_cfg = self.config.get("ollama", {})
        self._ollama_host = ollama_cfg.get("host", "http://localhost:11434")
        self._ollama_model = ollama_cfg.get("model", "llama3")
        seed = self.config.get("seed", 42)
        self._ollama_options = {
            **get_fixed_ollama_options(seed),
            **ollama_cfg.get("options", {}),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_case: dict[str, Any],
        actual_response: str,
    ) -> EvalResult:
        reference = test_case.get("expected_response", "")
        if not reference:
            return EvalResult(
                metric=self.name,
                score=1.0,
                details={"note": "No reference — semantic similarity skipped"},
            )

        hyp = normalize_text(actual_response)
        ref = normalize_text(reference)

        details: dict[str, Any] = {}

        # 1. Embedding cosine similarity
        cos_sim = self._embedding_cosine(ref, hyp)
        details["cosine_similarity"] = round(cos_sim, 6)
        details["embedding_model"] = self._embedding_model_name

        # 2. LLM judge (optional)
        llm_score_norm: float | None = None
        if self._llm_judge:
            llm_raw = self._llm_judge_score(ref, hyp, test_case.get("id", "?"))
            if llm_raw is not None:
                llm_score_norm = llm_raw / 10.0
                details["llm_judge_raw"] = llm_raw
                details["llm_judge_score"] = round(llm_score_norm, 6)

        # 3. Combine
        if llm_score_norm is not None:
            combined = (
                self._emb_weight * cos_sim + self._llm_weight * llm_score_norm
            )
            details["combined_weights"] = {
                "embedding": self._emb_weight,
                "llm_judge": self._llm_weight,
            }
        else:
            combined = cos_sim
            details["combined_weights"] = {"embedding": 1.0, "llm_judge": 0.0}

        score = round(max(0.0, min(1.0, combined)), 6)

        logger.debug(
            f"[{test_case.get('id')}] cos={cos_sim:.4f} "
            f"llm={llm_score_norm} combined={score:.4f}"
        )

        return EvalResult(
            metric=self.name,
            score=score,
            raw_value=score,
            details=details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _embedding_cosine(self, ref: str, hyp: str) -> float:
        """Compute cosine similarity between sentence embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        model = self._get_embedding_model()
        # encode returns (1, dim) shaped arrays
        emb_ref = model.encode([ref], normalize_embeddings=True)
        emb_hyp = model.encode([hyp], normalize_embeddings=True)

        sim = float(cosine_similarity(emb_ref, emb_hyp)[0][0])
        # Clip to [0, 1] — cosine can be slightly negative for unrelated text
        return max(0.0, sim)

    def _llm_judge_score(
        self,
        ref: str,
        hyp: str,
        case_id: str,
    ) -> float | None:
        """
        Ask the Ollama LLM to rate semantic equivalence (0-10).
        Returns None on failure so the embedding score is used alone.
        """
        try:
            import ollama

            prompt = _SEMANTIC_JUDGE_PROMPT.format(
                reference=ref[:1000],  # Truncate to avoid context overflow
                hypothesis=hyp[:1000],
            )

            response = ollama.generate(
                model=self._ollama_model,
                prompt=prompt,
                options=self._ollama_options,
                host=self._ollama_host,
            )

            raw_text = response.get("response", "").strip()
            return self._parse_score(raw_text, case_id)

        except Exception as exc:
            logger.warning(
                f"[{case_id}] Ollama LLM judge failed: {exc} "
                "— falling back to embedding-only score"
            )
            return None

    @staticmethod
    def _parse_score(text: str, case_id: str) -> float | None:
        """Extract integer 0-10 from LLM response."""
        # Match first integer in response
        match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 10:
                return float(val)
        logger.warning(f"[{case_id}] Could not parse LLM judge score from: {text!r}")
        return None
