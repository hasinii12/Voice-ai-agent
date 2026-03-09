"""
JSON report generator for evaluation runs.

Produces structured, machine-readable reports with full audit trail
including input hashes, config hashes, and per-case breakdowns.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for type-safe report construction
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    """Single metric score for one test case."""

    metric: str
    score: float
    raw_value: float | None = None
    unit: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CaseReport(BaseModel):
    """Full evaluation result for one test case."""

    case_id: str
    category: str
    input_text: str
    input_hash: str
    expected_response: str
    actual_response: str
    transcription: str | None = None
    latency_ms: float | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    composite_score: float | None = None
    passed: bool = False
    error: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SummaryStats(BaseModel):
    """Aggregate statistics across all test cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    pass_rate: float
    avg_wer: float | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    avg_semantic_similarity: float | None = None
    avg_hallucination_rate: float | None = None
    avg_composite_score: float | None = None


class EvaluationReport(BaseModel):
    """Top-level evaluation report."""

    run_id: str
    config_hash: str
    seed: int
    timestamp: str
    duration_seconds: float
    whisper_model: str
    ollama_model: str
    summary: SummaryStats
    thresholds: dict[str, Any]
    cases: list[CaseReport]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Builds and persists evaluation reports."""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_report(
        self,
        run_id: str,
        config: dict[str, Any],
        config_hash: str,
        seed: int,
        start_time: datetime,
        end_time: datetime,
        case_reports: list[CaseReport],
        thresholds: dict[str, Any],
    ) -> EvaluationReport:
        """Assemble the full evaluation report from case results."""

        duration = (end_time - start_time).total_seconds()

        summary = self._compute_summary(case_reports, thresholds)

        report = EvaluationReport(
            run_id=run_id,
            config_hash=config_hash,
            seed=seed,
            timestamp=start_time.isoformat(),
            duration_seconds=round(duration, 3),
            whisper_model=config.get("whisper", {}).get("model_size", "unknown"),
            ollama_model=config.get("ollama", {}).get("model", "unknown"),
            summary=summary,
            thresholds=thresholds,
            cases=case_reports,
            metadata={
                "framework_version": "1.0.0",
                "python_version": self._python_version(),
            },
        )

        return report

    def save_report(self, report: EvaluationReport) -> Path:
        """Write report to JSON file. Returns the output path."""
        filename = f"{report.run_id}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                report.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        logger.info(f"Report saved → {output_path}")
        return output_path

    def save_summary(self, report: EvaluationReport) -> Path:
        """Save a lightweight summary-only JSON (no per-case details)."""
        filename = f"{report.run_id}_summary.json"
        output_path = self.output_dir / filename

        summary_data = {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "duration_seconds": report.duration_seconds,
            "seed": report.seed,
            "config_hash": report.config_hash,
            "summary": report.summary.model_dump(),
            "thresholds": report.thresholds,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        cases: list[CaseReport],
        thresholds: dict[str, Any],
    ) -> SummaryStats:
        """Aggregate per-case metrics into summary statistics."""
        import numpy as np

        total = len(cases)
        error_cases = sum(1 for c in cases if c.error)
        passed = sum(1 for c in cases if c.passed)
        failed = total - passed - error_cases

        # Extract per-metric scores
        def get_scores(metric_name: str) -> list[float]:
            scores = []
            for case in cases:
                for m in case.metrics:
                    if m.metric == metric_name and m.error is None:
                        scores.append(m.score)
            return scores

        wer_scores = get_scores("wer")
        semantic_scores = get_scores("semantic_similarity")
        hallucination_scores = get_scores("hallucination")
        composite_scores = [c.composite_score for c in cases if c.composite_score is not None]

        # Latency from case reports directly
        latencies = [c.latency_ms for c in cases if c.latency_ms is not None]

        def safe_mean(lst: list[float]) -> float | None:
            return round(float(np.mean(lst)), 4) if lst else None

        def safe_percentile(lst: list[float], p: int) -> float | None:
            return round(float(np.percentile(lst, p)), 2) if lst else None

        # WER: lower is better → report as-is
        # Hallucination: lower is better → stored as rate (0=no hallucination)
        return SummaryStats(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=error_cases,
            pass_rate=round(passed / total, 4) if total > 0 else 0.0,
            avg_wer=safe_mean(wer_scores),
            avg_latency_ms=safe_mean(latencies),
            p50_latency_ms=safe_percentile(latencies, 50),
            p90_latency_ms=safe_percentile(latencies, 90),
            p95_latency_ms=safe_percentile(latencies, 95),
            p99_latency_ms=safe_percentile(latencies, 99),
            avg_semantic_similarity=safe_mean(semantic_scores),
            avg_hallucination_rate=safe_mean(hallucination_scores),
            avg_composite_score=safe_mean(composite_scores),
        )

    @staticmethod
    def _python_version() -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
