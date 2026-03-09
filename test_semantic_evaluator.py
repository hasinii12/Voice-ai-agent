"""
Tests for Semantic Similarity Evaluator.

Uses mocked Ollama to avoid requiring a running LLM server in CI.
The embedding tests require sentence-transformers (install separately).
"""

import pytest
from unittest.mock import patch, MagicMock
from evaluators.semantic_evaluator import SemanticEvaluator
from utils.determinism import seed_everything


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def evaluator_no_llm():
    """Evaluator with LLM judge disabled — embedding only."""
    return SemanticEvaluator(config={"llm_judge": False})


@pytest.fixture
def make_case():
    def _make(expected: str, input_text: str = "test question"):
        return {
            "id": "sem_001",
            "input_text": input_text,
            "expected_response": expected,
            "reference_facts": [],
        }
    return _make


def _mock_ollama_generate(score: int):
    """Factory for mock ollama.generate returning a specific score."""
    mock_resp = {"response": str(score)}
    return MagicMock(return_value=mock_resp)


class TestSemanticEvaluatorEmbedding:

    @pytest.mark.parametrize("text", [
        "The capital of France is Paris.",
        "Hello world",
    ])
    def test_identical_texts_score_one(self, evaluator_no_llm, make_case, text):
        tc = make_case(text)
        result = evaluator_no_llm.evaluate(tc, text)
        assert result.score >= 0.99, f"Expected ~1.0 for identical text, got {result.score}"

    def test_unrelated_texts_score_low(self, evaluator_no_llm, make_case):
        tc = make_case("The capital of France is Paris.")
        result = evaluator_no_llm.evaluate(
            tc, "Quantum mechanics describes subatomic particle behavior."
        )
        assert result.score < 0.7

    def test_semantically_similar_texts_score_high(self, evaluator_no_llm, make_case):
        tc = make_case("Paris is the capital city of France.")
        result = evaluator_no_llm.evaluate(
            tc, "France's capital is the city of Paris."
        )
        assert result.score > 0.8

    def test_no_reference_returns_perfect_score(self, evaluator_no_llm):
        tc = {"id": "t", "input_text": "q", "expected_response": "", "reference_facts": []}
        result = evaluator_no_llm.evaluate(tc, "some response")
        assert result.score == pytest.approx(1.0)

    def test_score_in_valid_range(self, evaluator_no_llm, make_case):
        tc = make_case("Test sentence one.")
        result = evaluator_no_llm.evaluate(tc, "Completely different content here.")
        assert 0.0 <= result.score <= 1.0

    def test_cosine_similarity_in_details(self, evaluator_no_llm, make_case):
        tc = make_case("Hello world")
        result = evaluator_no_llm.evaluate(tc, "Hello world")
        assert "cosine_similarity" in result.details
        assert "embedding_model" in result.details

    def test_metric_name(self, evaluator_no_llm, make_case):
        tc = make_case("test")
        result = evaluator_no_llm.evaluate(tc, "test")
        assert result.metric == "semantic_similarity"

    def test_deterministic_embedding_score(self, evaluator_no_llm, make_case):
        """Same input must produce identical embedding score."""
        tc = make_case("The sky is blue.")
        responses = ["The sky appears blue in color."] * 5
        scores = [evaluator_no_llm.evaluate(tc, r).score for r in responses]
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"


class TestSemanticEvaluatorWithLLMJudge:

    @patch("evaluators.semantic_evaluator.ollama")
    def test_llm_judge_combined_score(self, mock_ollama, make_case):
        mock_ollama.generate.return_value = {"response": "9"}
        evaluator = SemanticEvaluator(config={
            "llm_judge": True,
            "llm_judge_weight": 0.4,
            "embedding_weight": 0.6,
        })
        tc = make_case("Paris is the capital of France.")
        result = evaluator.evaluate(tc, "France's capital is Paris.")
        assert result.score > 0.8
        assert "llm_judge_raw" in result.details
        assert result.details["llm_judge_raw"] == 9.0

    @patch("evaluators.semantic_evaluator.ollama")
    def test_llm_judge_failure_falls_back_to_embedding(self, mock_ollama, make_case):
        mock_ollama.generate.side_effect = ConnectionError("Ollama not running")
        evaluator = SemanticEvaluator(config={"llm_judge": True})
        tc = make_case("Paris is the capital of France.")
        result = evaluator.evaluate(tc, "Paris is the capital of France.")
        # Should still return a score (from embedding only)
        assert result.score > 0.9
        assert "llm_judge_raw" not in result.details

    @patch("evaluators.semantic_evaluator.ollama")
    def test_llm_unparseable_output_falls_back(self, mock_ollama, make_case):
        mock_ollama.generate.return_value = {"response": "I cannot rate this."}
        evaluator = SemanticEvaluator(config={"llm_judge": True})
        tc = make_case("test")
        result = evaluator.evaluate(tc, "test")
        # Should not raise, should return embedding score
        assert 0.0 <= result.score <= 1.0


class TestSemanticParseScore:

    def test_parse_valid_integer(self):
        from evaluators.semantic_evaluator import SemanticEvaluator
        e = SemanticEvaluator.__new__(SemanticEvaluator)
        assert SemanticEvaluator._parse_score("8", "tc1") == 8.0
        assert SemanticEvaluator._parse_score("Score: 7", "tc1") == 7.0
        assert SemanticEvaluator._parse_score("0", "tc1") == 0.0
        assert SemanticEvaluator._parse_score("10", "tc1") == 10.0

    def test_parse_invalid_returns_none(self):
        from evaluators.semantic_evaluator import SemanticEvaluator
        assert SemanticEvaluator._parse_score("", "tc1") is None
        assert SemanticEvaluator._parse_score("abc", "tc1") is None
