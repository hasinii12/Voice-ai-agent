"""
Word Error Rate (WER) Evaluator.

WER measures transcription accuracy by computing the minimum edit distance
(insertions, deletions, substitutions) between hypothesis and reference
word sequences, normalised by the reference length.

WER = (S + D + I) / N
  S = substitutions, D = deletions, I = insertions, N = reference words

Score returned is (1 - WER), clipped to [0, 1], so higher = better.

Determinism: WER is a pure function of normalised text strings.
No randomness involved. The normalisation pipeline is fixed.
"""

from typing import Any

from jiwer import wer as compute_wer
from jiwer import cer as compute_cer

from evaluators.base_evaluator import BaseEvaluator, EvalResult
from utils.determinism import normalize_text
from utils.logger import get_logger

logger = get_logger(__name__)


class WERevaluator(BaseEvaluator):
    """
    Evaluates Word Error Rate between the Voice AI response and the
    expected response (or audio transcription reference).

    Config keys
    -----------
    normalize.lowercase            : bool (default True)
    normalize.remove_punctuation   : bool (default True)
    normalize.expand_contractions  : bool (default True)
    normalize.strip_extra_whitespace: bool (default True)
    compute_cer                    : bool (default False) — also compute CER
    """

    name = "wer"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        norm_cfg = self.config.get("normalize", {})
        self._norm_kwargs = {
            "lowercase": norm_cfg.get("lowercase", True),
            "remove_punctuation": norm_cfg.get("remove_punctuation", True),
            "expand_contractions": norm_cfg.get("expand_contractions", True),
            "strip_extra_whitespace": norm_cfg.get("strip_extra_whitespace", True),
        }
        self._compute_cer = self.config.get("compute_cer", False)

    def evaluate(
        self,
        test_case: dict[str, Any],
        actual_response: str,
    ) -> EvalResult:
        """
        Compute WER between actual_response and expected_response.

        If the test case includes a 'transcription_reference' field
        (ground-truth transcript for audio input), that is used instead
        of 'expected_response' for WER computation.
        """
        reference_text = test_case.get(
            "transcription_reference",
            test_case.get("expected_response", ""),
        )

        if not reference_text:
            return EvalResult(
                metric=self.name,
                score=1.0,
                raw_value=0.0,
                unit="wer",
                details={"note": "No reference text — WER skipped"},
            )

        # Normalise both strings identically
        hyp = normalize_text(actual_response, **self._norm_kwargs)
        ref = normalize_text(reference_text, **self._norm_kwargs)

        if not ref:
            logger.warning(f"[{test_case.get('id')}] Reference empty after normalisation")
            return EvalResult(
                metric=self.name,
                score=1.0,
                raw_value=0.0,
                unit="wer",
                details={"note": "Reference empty after normalisation"},
            )

        # jiwer handles empty hypothesis gracefully
        raw_wer = compute_wer(ref, hyp)
        # Clip to [0, 1] — WER can exceed 1.0 for very bad hypotheses
        raw_wer = min(raw_wer, 1.0)

        details: dict[str, Any] = {
            "reference_normalised": ref,
            "hypothesis_normalised": hyp,
            "reference_word_count": len(ref.split()),
            "hypothesis_word_count": len(hyp.split()),
        }

        if self._compute_cer:
            raw_cer = min(compute_cer(ref, hyp), 1.0)
            details["cer"] = round(raw_cer, 6)

        # Score: 1 - WER (higher = better match)
        score = max(0.0, 1.0 - raw_wer)

        logger.debug(
            f"[{test_case.get('id')}] WER={raw_wer:.4f} score={score:.4f}"
        )

        return EvalResult(
            metric=self.name,
            score=round(score, 6),
            raw_value=round(raw_wer, 6),
            unit="wer",
            details=details,
        )
