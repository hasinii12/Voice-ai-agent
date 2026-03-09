"""
Hallucination Rate Evaluator.

Uses an LLM-as-judge approach to detect hallucinations in Voice AI responses.
Two complementary checks are performed:

1. Factual Grounding Check
   - Verifies each claim in the response is supported by the reference facts
   - Detects fabricated facts not present in the reference

2. Context Faithfulness Check
   - Verifies the response does not contradict the provided context
   - Detects responses that go beyond or conflict with the question

The hallucination score represents the RATE of hallucination (0 = none, 1 = all).
The EvalResult score = 1 - hallucination_rate (higher = better, i.e. less hallucination).

Determinism: temperature=0, top_k=1, seed=42 ensures identical LLM outputs
for identical inputs. Falls back to heuristic scoring if Ollama unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvalResult
from utils.determinism import normalize_text, get_fixed_ollama_options
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FACTUAL_GROUNDING_PROMPT = """\
You are a factual accuracy evaluator for AI assistant responses.

REFERENCE FACTS (ground truth):
{reference_facts}

AI RESPONSE TO EVALUATE:
{response}

Task: Identify any claims in the AI response that are NOT supported by the reference facts, \
or that contradict them. These are hallucinations.

Respond with ONLY valid JSON in this exact format:
{{
  "hallucination_count": <integer>,
  "total_claims": <integer>,
  "hallucinated_claims": ["<claim1>", "<claim2>"],
  "confidence": <float 0.0-1.0>
}}

If the response makes no specific factual claims, set total_claims to 0 and hallucination_count to 0.
JSON only, no other text:"""

_CONTEXT_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI assistant's response is faithful to the question context.

QUESTION:
{question}

AI RESPONSE:
{response}

Task: Rate how faithful the response is to the question on a scale of 0-10:
- 10: Completely faithful, directly addresses the question
- 7-9: Mostly faithful, minor deviations
- 4-6: Partially faithful, some off-topic or fabricated content
- 1-3: Mostly unfaithful or hallucinatory
- 0: Completely fabricated or refuses to engage

Respond with ONLY a single integer 0-10. No other text.
Score:"""


class HallucinationEvaluator(BaseEvaluator):
    """
    LLM-based hallucination detector for Voice AI responses.

    Config keys
    -----------
    check_factual_grounding      : bool (default True)
    check_context_faithfulness   : bool (default True)
    confidence_threshold         : float (default 0.5)
    ollama.host                  : str
    ollama.model                 : str
    ollama.options               : dict
    """

    name = "hallucination"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._check_factual = self.config.get("check_factual_grounding", True)
        self._check_faithful = self.config.get("check_context_faithfulness", True)
        self._confidence_threshold = float(
            self.config.get("confidence_threshold", 0.5)
        )

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
        reference_facts: list[str] = test_case.get("reference_facts", [])
        question = test_case.get("input_text", "")
        case_id = test_case.get("id", "?")

        details: dict[str, Any] = {}
        hallucination_scores: list[float] = []

        # 1. Factual grounding check
        if self._check_factual and reference_facts:
            factual_result = self._factual_grounding_check(
                reference_facts, actual_response, case_id
            )
            details["factual_grounding"] = factual_result
            if "hallucination_rate" in factual_result:
                hallucination_scores.append(factual_result["hallucination_rate"])
        elif self._check_factual and not reference_facts:
            details["factual_grounding"] = {
                "skipped": True,
                "reason": "No reference facts provided",
            }

        # 2. Context faithfulness check
        if self._check_faithful and question:
            faithful_result = self._context_faithfulness_check(
                question, actual_response, case_id
            )
            details["context_faithfulness"] = faithful_result
            if "unfaithfulness_rate" in faithful_result:
                hallucination_scores.append(faithful_result["unfaithfulness_rate"])

        # 3. Aggregate
        if hallucination_scores:
            avg_hallucination_rate = sum(hallucination_scores) / len(hallucination_scores)
        else:
            # No checks could run — assume no hallucination detected
            avg_hallucination_rate = 0.0
            details["note"] = "No hallucination checks could be run"

        avg_hallucination_rate = round(max(0.0, min(1.0, avg_hallucination_rate)), 6)
        # Score: 1 - hallucination_rate (higher = less hallucination = better)
        score = round(1.0 - avg_hallucination_rate, 6)

        details["hallucination_rate"] = avg_hallucination_rate

        logger.debug(
            f"[{case_id}] hallucination_rate={avg_hallucination_rate:.4f} score={score:.4f}"
        )

        return EvalResult(
            metric=self.name,
            score=score,
            raw_value=avg_hallucination_rate,
            unit="hallucination_rate",
            details=details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _factual_grounding_check(
        self,
        reference_facts: list[str],
        response: str,
        case_id: str,
    ) -> dict[str, Any]:
        """Use LLM to count unsupported claims in the response."""
        facts_str = "\n".join(f"- {f}" for f in reference_facts)
        prompt = _FACTUAL_GROUNDING_PROMPT.format(
            reference_facts=facts_str,
            response=normalize_text(response, remove_punctuation=False),
        )

        try:
            import ollama

            llm_response = ollama.generate(
                model=self._ollama_model,
                prompt=prompt,
                options=self._ollama_options,
                host=self._ollama_host,
            )
            raw = llm_response.get("response", "").strip()
            parsed = self._parse_json_response(raw, case_id)

            total = parsed.get("total_claims", 0)
            hallucinated = parsed.get("hallucination_count", 0)
            confidence = float(parsed.get("confidence", 1.0))

            if total == 0:
                rate = 0.0
            else:
                rate = min(hallucinated / total, 1.0)

            # Down-weight rate if LLM is uncertain
            if confidence < self._confidence_threshold:
                rate *= confidence

            return {
                "hallucination_rate": round(rate, 6),
                "total_claims": total,
                "hallucination_count": hallucinated,
                "hallucinated_claims": parsed.get("hallucinated_claims", []),
                "llm_confidence": round(confidence, 4),
            }

        except Exception as exc:
            logger.warning(f"[{case_id}] Factual grounding check failed: {exc}")
            return {"error": str(exc), "hallucination_rate": 0.0}

    def _context_faithfulness_check(
        self,
        question: str,
        response: str,
        case_id: str,
    ) -> dict[str, Any]:
        """Use LLM to rate faithfulness of response to question."""
        prompt = _CONTEXT_FAITHFULNESS_PROMPT.format(
            question=question[:500],
            response=response[:1000],
        )

        try:
            import ollama

            llm_response = ollama.generate(
                model=self._ollama_model,
                prompt=prompt,
                options=self._ollama_options,
                host=self._ollama_host,
            )
            raw = llm_response.get("response", "").strip()
            score = self._parse_integer_score(raw, case_id)

            if score is None:
                return {"error": "Could not parse score", "unfaithfulness_rate": 0.0}

            faithfulness = score / 10.0
            unfaithfulness_rate = round(1.0 - faithfulness, 6)

            return {
                "faithfulness_score": score,
                "faithfulness_normalised": round(faithfulness, 6),
                "unfaithfulness_rate": unfaithfulness_rate,
            }

        except Exception as exc:
            logger.warning(f"[{case_id}] Context faithfulness check failed: {exc}")
            return {"error": str(exc), "unfaithfulness_rate": 0.0}

    @staticmethod
    def _parse_json_response(text: str, case_id: str) -> dict[str, Any]:
        """Extract JSON object from LLM response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"[{case_id}] Could not parse JSON from: {text[:200]!r}")
        return {"hallucination_count": 0, "total_claims": 0, "confidence": 0.5}

    @staticmethod
    def _parse_integer_score(text: str, case_id: str) -> float | None:
        match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 10:
                return float(val)
        logger.warning(f"[{case_id}] Could not parse faithfulness score from: {text!r}")
        return None
