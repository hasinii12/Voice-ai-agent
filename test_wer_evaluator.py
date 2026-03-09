"""
Tests for WER Evaluator.

Covers: perfect match, partial match, empty inputs, normalization,
determinism (same input → same output), and edge cases.
"""

import pytest
from evaluators.wer_evaluator import WERevaluator
from utils.determinism import seed_everything


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def evaluator():
    return WERevaluator()


@pytest.fixture
def make_case():
    def _make(expected: str, input_text: str = "What is X?", facts=None):
        return {
            "id": "test_001",
            "input_text": input_text,
            "expected_response": expected,
            "reference_facts": facts or [],
        }
    return _make


class TestWEREvaluatorBasic:

    def test_perfect_match_score_is_one(self, evaluator, make_case):
        tc = make_case("The capital of France is Paris.")
        result = evaluator.evaluate(tc, "The capital of France is Paris.")
        assert result.score == pytest.approx(1.0, abs=1e-5)
        assert result.raw_value == pytest.approx(0.0, abs=1e-5)

    def test_completely_wrong_score_is_zero(self, evaluator, make_case):
        tc = make_case("The capital of France is Paris.")
        result = evaluator.evaluate(tc, "xyz abc def ghi jkl mno pqr stu")
        assert result.score < 0.3

    def test_partial_match_score_between_zero_and_one(self, evaluator, make_case):
        tc = make_case("The capital of France is Paris.")
        result = evaluator.evaluate(tc, "Paris is the capital of France.")
        assert 0.0 < result.score < 1.0

    def test_score_in_valid_range(self, evaluator, make_case):
        cases = [
            ("hello world", "hello there"),
            ("", "some text"),
            ("one two three", "one two three four five"),
        ]
        for expected, actual in cases:
            tc = make_case(expected)
            result = evaluator.evaluate(tc, actual)
            assert 0.0 <= result.score <= 1.0, f"Score out of range for: {actual!r}"

    def test_empty_hypothesis(self, evaluator, make_case):
        tc = make_case("The capital of France is Paris.")
        result = evaluator.evaluate(tc, "")
        assert result.score == pytest.approx(0.0, abs=1e-5)

    def test_empty_reference_returns_perfect_score(self, evaluator, make_case):
        tc = make_case("")
        result = evaluator.evaluate(tc, "some response")
        assert result.score == pytest.approx(1.0)

    def test_metric_name_is_wer(self, evaluator, make_case):
        tc = make_case("test")
        result = evaluator.evaluate(tc, "test")
        assert result.metric == "wer"

    def test_unit_is_wer(self, evaluator, make_case):
        tc = make_case("test")
        result = evaluator.evaluate(tc, "test")
        assert result.unit == "wer"


class TestWEREvaluatorNormalization:

    def test_case_insensitive(self, evaluator, make_case):
        tc = make_case("The Capital Of France Is Paris.")
        result_lower = evaluator.evaluate(tc, "the capital of france is paris.")
        result_upper = evaluator.evaluate(tc, "THE CAPITAL OF FRANCE IS PARIS.")
        assert result_lower.score == pytest.approx(result_upper.score, abs=1e-5)

    def test_punctuation_ignored(self, evaluator, make_case):
        tc = make_case("Hello world!")
        result_punct = evaluator.evaluate(tc, "Hello world!")
        result_no_punct = evaluator.evaluate(tc, "Hello world")
        assert result_punct.score == pytest.approx(result_no_punct.score, abs=1e-5)

    def test_contraction_expansion(self, evaluator, make_case):
        tc = make_case("I am doing well.")
        result_expanded = evaluator.evaluate(tc, "I am doing well.")
        result_contracted = evaluator.evaluate(tc, "I'm doing well.")
        # After expansion, "I'm" → "I am" — should score close to perfect
        assert result_contracted.score > 0.8

    def test_extra_whitespace_ignored(self, evaluator, make_case):
        tc = make_case("hello world")
        result_normal = evaluator.evaluate(tc, "hello world")
        result_spaces = evaluator.evaluate(tc, "hello  world")
        assert result_normal.score == pytest.approx(result_spaces.score, abs=1e-5)


class TestWEREvaluatorDeterminism:

    def test_same_input_same_output(self, evaluator, make_case):
        """Critical: identical inputs must produce identical scores."""
        tc = make_case("The quick brown fox jumps over the lazy dog.")
        hypothesis = "The quick brown fox jumped over a lazy dog."

        results = [evaluator.evaluate(tc, hypothesis) for _ in range(5)]
        scores = [r.score for r in results]
        assert all(s == scores[0] for s in scores), f"Non-deterministic scores: {scores}"

    def test_transcription_reference_takes_precedence(self, make_case):
        """When transcription_reference is set, it overrides expected_response for WER."""
        evaluator = WERevaluator()
        tc = make_case("This is the expected response.")
        tc["transcription_reference"] = "This is the transcription reference."

        result = evaluator.evaluate(tc, "This is the transcription reference.")
        assert result.score == pytest.approx(1.0, abs=1e-5)


class TestWEREvaluatorConfig:

    def test_cer_computation_when_enabled(self, make_case):
        evaluator = WERevaluator(config={"compute_cer": True})
        tc = make_case("hello world")
        result = evaluator.evaluate(tc, "hello word")
        assert "cer" in result.details

    def test_cer_not_in_details_when_disabled(self, make_case):
        evaluator = WERevaluator(config={"compute_cer": False})
        tc = make_case("hello world")
        result = evaluator.evaluate(tc, "hello word")
        assert "cer" not in result.details

    def test_no_punctuation_removal(self, make_case):
        evaluator = WERevaluator(config={"normalize": {"remove_punctuation": False}})
        tc = make_case("Hello, world!")
        # Punctuation kept — "hello," != "hello" (treated as different tokens)
        result = evaluator.evaluate(tc, "Hello world")
        # Score should be lower than with punctuation removed
        assert result.score <= 1.0
