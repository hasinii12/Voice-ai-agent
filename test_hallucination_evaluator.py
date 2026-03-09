"""
Tests for Hallucination Evaluator.

All LLM calls are mocked so tests run without Ollama.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from evaluators.hallucination_evaluator import HallucinationEvaluator
from utils.determinism import seed_everything


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def evaluator():
    return HallucinationEvaluator()


def make_case(
    facts: list[str],
    input_text: str = "Tell me about Paris.",
) -> dict:
    return {
        "id": "hall_001",
        "input_text": input_text,
        "expected_response": "Paris is the capital of France.",
        "reference_facts": facts,
    }


def _mock_ollama_factual(hallucination_count: int, total_claims: int, confidence: float = 1.0):
    payload = json.dumps({
        "hallucination_count": hallucination_count,
        "total_claims": total_claims,
        "hallucinated_claims": [f"claim_{i}" for i in range(hallucination_count)],
        "confidence": confidence,
    })
    return {"response": payload}


def _mock_ollama_faithful(score: int):
    return {"response": str(score)}


class TestHallucinationEvaluatorFactualGrounding:

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_no_hallucinations_score_is_one(self, mock_ollama):
        mock_ollama.generate.return_value = _mock_ollama_factual(0, 3)
        evaluator = HallucinationEvaluator(config={"check_context_faithfulness": False})
        tc = make_case(["Paris is the capital of France", "Paris is in Europe"])
        result = evaluator.evaluate(tc, "Paris is the capital of France and is in Europe.")
        assert result.score == pytest.approx(1.0, abs=1e-3)
        assert result.raw_value == pytest.approx(0.0, abs=1e-3)

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_full_hallucination_score_is_zero(self, mock_ollama):
        mock_ollama.generate.return_value = _mock_ollama_factual(3, 3)
        evaluator = HallucinationEvaluator(config={"check_context_faithfulness": False})
        tc = make_case(["Paris is small", "Paris has no museums"])
        result = evaluator.evaluate(tc, "Paris has a population of 100.")
        assert result.score == pytest.approx(0.0, abs=1e-3)

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_partial_hallucination(self, mock_ollama):
        mock_ollama.generate.return_value = _mock_ollama_factual(1, 4)
        evaluator = HallucinationEvaluator(config={"check_context_faithfulness": False})
        tc = make_case(["fact1", "fact2", "fact3", "fact4"])
        result = evaluator.evaluate(tc, "response")
        expected_rate = 1 / 4  # 0.25
        expected_score = 1.0 - expected_rate
        assert result.score == pytest.approx(expected_score, abs=1e-3)

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_low_confidence_downweights_hallucination(self, mock_ollama):
        # 1 hallucination out of 2, but confidence=0.3 → rate * 0.3
        mock_ollama.generate.return_value = _mock_ollama_factual(1, 2, confidence=0.3)
        evaluator = HallucinationEvaluator(config={
            "check_context_faithfulness": False,
            "confidence_threshold": 0.5,
        })
        tc = make_case(["fact1", "fact2"])
        result = evaluator.evaluate(tc, "response")
        # rate = 0.5 * 0.3 = 0.15 → score = 0.85
        assert result.score > 0.8

    def test_no_facts_skips_factual_check(self):
        evaluator = HallucinationEvaluator(config={"check_context_faithfulness": False})
        tc = make_case(facts=[])
        result = evaluator.evaluate(tc, "some response")
        assert result.score == pytest.approx(1.0)
        assert result.details.get("factual_grounding", {}).get("skipped") is True


class TestHallucinationEvaluatorFaithfulness:

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_faithful_response_high_score(self, mock_ollama):
        mock_ollama.generate.return_value = _mock_ollama_faithful(10)
        evaluator = HallucinationEvaluator(config={"check_factual_grounding": False})
        tc = make_case(facts=[])
        result = evaluator.evaluate(tc, "A faithful response.")
        assert result.score == pytest.approx(1.0, abs=1e-3)

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_unfaithful_response_low_score(self, mock_ollama):
        mock_ollama.generate.return_value = _mock_ollama_faithful(0)
        evaluator = HallucinationEvaluator(config={"check_factual_grounding": False})
        tc = make_case(facts=[])
        result = evaluator.evaluate(tc, "Completely hallucinated response.")
        assert result.score == pytest.approx(0.0, abs=1e-3)


class TestHallucinationEvaluatorOllamaFailure:

    @patch("evaluators.hallucination_evaluator.ollama")
    def test_ollama_failure_returns_zero_hallucination(self, mock_ollama):
        mock_ollama.generate.side_effect = ConnectionError("Ollama offline")
        evaluator = HallucinationEvaluator()
        tc = make_case(["Paris is the capital of France"])
        result = evaluator.evaluate(tc, "Paris is the capital.")
        # Should not raise; should degrade gracefully
        assert 0.0 <= result.score <= 1.0

    def test_missing_input_text_skips_faithfulness(self):
        evaluator = HallucinationEvaluator(config={"check_factual_grounding": False})
        tc = {
            "id": "t",
            "input_text": "",
            "expected_response": "test",
            "reference_facts": [],
        }
        result = evaluator.evaluate(tc, "response")
        assert 0.0 <= result.score <= 1.0


class TestHallucinationJSONParsing:

    def test_parse_valid_json(self):
        from evaluators.hallucination_evaluator import HallucinationEvaluator
        raw = '{"hallucination_count": 1, "total_claims": 3, "confidence": 0.9}'
        parsed = HallucinationEvaluator._parse_json_response(raw, "t1")
        assert parsed["hallucination_count"] == 1
        assert parsed["total_claims"] == 3

    def test_parse_json_with_preamble(self):
        from evaluators.hallucination_evaluator import HallucinationEvaluator
        raw = 'Here is the result:\n{"hallucination_count": 0, "total_claims": 2, "confidence": 1.0}'
        parsed = HallucinationEvaluator._parse_json_response(raw, "t1")
        assert parsed["hallucination_count"] == 0

    def test_parse_unparseable_returns_defaults(self):
        from evaluators.hallucination_evaluator import HallucinationEvaluator
        raw = "I cannot evaluate this."
        parsed = HallucinationEvaluator._parse_json_response(raw, "t1")
        assert parsed.get("hallucination_count", 0) == 0
