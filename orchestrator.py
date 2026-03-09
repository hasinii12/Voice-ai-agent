"""
Evaluation Orchestrator — main entry point for the evaluation pipeline.

Responsibilities:
1. Load and validate configuration
2. Seed all RNG sources for determinism
3. Load test cases (from JSON or audio directory)
4. For each test case:
   a. Transcribe audio (if audio_file provided)
   b. Query the Voice AI system and record latency
   c. Run all enabled evaluators
   d. Compute composite score and pass/fail
5. Generate and save JSON report

Usage
-----
python -m pipeline.orchestrator --config configs/eval_config.yaml \
    --test-cases data/samples/sample_test_cases.json

python -m pipeline.orchestrator --config configs/eval_config.yaml \
    --audio-dir /path/to/audio/ --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from evaluators import (
    WERevaluator,
    LatencyEvaluator,
    SemanticEvaluator,
    HallucinationEvaluator,
)
from evaluators.base_evaluator import BaseEvaluator
from pipeline.audio_transcriber import AudioTranscriber, MockTranscriber
from pipeline.voice_ai_client import BaseVoiceAIClient, MockVoiceAIClient, OllamaVoiceAIClient
from utils.determinism import seed_everything, hash_config, hash_test_case
from utils.logger import get_logger, configure_root_logger
from utils.report_generator import CaseReport, MetricResult, ReportGenerator

load_dotenv()
console = Console()
logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates end-to-end evaluation of a Voice AI system.
    """

    def __init__(
        self,
        config: dict[str, Any],
        voice_ai_client: BaseVoiceAIClient | None = None,
        transcriber: AudioTranscriber | None = None,
        mock: bool = False,
    ):
        self.config = config
        self.mock = mock

        # Seed everything FIRST before any other initialisation
        seed = config.get("evaluation", {}).get("seed", 42)
        seed_everything(seed)
        self.seed = seed

        # Components
        self.transcriber = transcriber or self._build_transcriber(mock)
        self.voice_ai_client = voice_ai_client or self._build_client(mock)
        self.evaluators = self._build_evaluators()
        self.report_gen = ReportGenerator(
            config.get("evaluation", {}).get("output_dir", "./reports")
        )

        self.config_hash = hash_config(config)
        logger.info(f"Orchestrator ready | seed={seed} config_hash={self.config_hash[:12]}...")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        test_cases: list[dict[str, Any]],
    ) -> Path:
        """
        Run the full evaluation pipeline on the provided test cases.
        Returns the path to the saved JSON report.
        """
        prefix = self.config.get("evaluation", {}).get("run_id_prefix", "eval")
        run_id = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)

        console.rule(f"[bold blue]Voice AI Evaluation Run: {run_id}")
        console.print(f"  Seed: {self.seed}  |  Cases: {len(test_cases)}  |  Mock: {self.mock}")
        console.print(f"  Evaluators: {[e.name for e in self.evaluators]}\n")

        case_reports: list[CaseReport] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(test_cases))

            for tc in test_cases:
                progress.update(task, description=f"[cyan]{tc.get('id', '?')}")
                report = self._evaluate_case(tc)
                case_reports.append(report)
                progress.advance(task)

        end_time = datetime.now(timezone.utc)

        thresholds = self.config.get("thresholds", {})
        report = self.report_gen.build_report(
            run_id=run_id,
            config=self.config,
            config_hash=self.config_hash,
            seed=self.seed,
            start_time=start_time,
            end_time=end_time,
            case_reports=case_reports,
            thresholds=thresholds,
        )

        report_path = self.report_gen.save_report(report)
        self.report_gen.save_summary(report)

        self._print_summary(report.summary)
        return report_path

    # ------------------------------------------------------------------
    # Per-case evaluation
    # ------------------------------------------------------------------

    def _evaluate_case(self, test_case: dict[str, Any]) -> CaseReport:
        """Run the full evaluation pipeline for a single test case."""
        case_id = test_case.get("id", "unknown")
        input_text = test_case.get("input_text", "")
        expected = test_case.get("expected_response", "")

        input_hash = hash_test_case(test_case)
        transcription: str | None = None

        try:
            # Step 1: Transcribe audio if provided
            audio_file = test_case.get("audio_file")
            if audio_file and Path(audio_file).exists():
                t_result = self.transcriber.transcribe(audio_file)
                transcription = t_result.text
                # Use transcription as the actual input for the Voice AI
                input_text = transcription
                logger.debug(f"[{case_id}] Transcription: {transcription[:80]}...")

            # Step 2: Query the Voice AI system
            ai_response = self.voice_ai_client.query(
                input_text,
                _expected_response=expected,  # Used by MockClient only
            )

            # Step 3: Inject timing data into test_case for latency evaluator
            test_case["_latency_ms"] = ai_response.latency_ms
            test_case["_ttfb_ms"] = ai_response.ttfb_ms

            actual_response = ai_response.text
            latency_ms = ai_response.latency_ms

            if ai_response.error:
                logger.warning(f"[{case_id}] Voice AI error: {ai_response.error}")

            # Step 4: Run all evaluators
            metric_results: list[MetricResult] = []
            for evaluator in self.evaluators:
                result = evaluator.safe_evaluate(test_case, actual_response)
                metric_results.append(result.to_metric_result())

            # Step 5: Compute composite score
            composite = self._composite_score(metric_results)

            # Step 6: Determine pass/fail
            passed = self._check_pass(composite, metric_results)

            return CaseReport(
                case_id=case_id,
                category=test_case.get("category", "unknown"),
                input_text=test_case.get("input_text", input_text),
                input_hash=input_hash,
                expected_response=expected,
                actual_response=actual_response,
                transcription=transcription,
                latency_ms=latency_ms,
                metrics=metric_results,
                composite_score=composite,
                passed=passed,
            )

        except Exception as exc:
            logger.error(f"[{case_id}] Evaluation failed: {exc}", exc_info=True)
            return CaseReport(
                case_id=case_id,
                category=test_case.get("category", "unknown"),
                input_text=test_case.get("input_text", input_text),
                input_hash=input_hash,
                expected_response=expected,
                actual_response="",
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _composite_score(self, metrics: list[MetricResult]) -> float:
        """
        Compute weighted composite score from individual metric scores.

        Weights are normalised so they always sum to 1, even if some
        metrics are missing (e.g. Ollama unavailable → no hallucination score).
        """
        weights_cfg = self.config.get("scoring", {}).get("weights", {})
        default_weights = {
            "wer": 0.25,
            "latency": 0.20,
            "semantic_similarity": 0.30,
            "hallucination": 0.25,
        }
        weights = {**default_weights, **weights_cfg}

        scored = {m.metric: m.score for m in metrics if m.error is None}
        if not scored:
            return 0.0

        # Only use weights for metrics that were actually computed
        applicable = {k: v for k, v in weights.items() if k in scored}
        if not applicable:
            return sum(scored.values()) / len(scored)

        total_weight = sum(applicable.values())
        composite = sum(
            scored[k] * (v / total_weight) for k, v in applicable.items()
        )
        return round(composite, 6)

    def _check_pass(
        self, composite: float, metrics: list[MetricResult]
    ) -> bool:
        """Check if the case passes all configured thresholds."""
        t = self.config.get("thresholds", {})
        min_composite = t.get("min_composite_score", 0.70)

        if composite < min_composite:
            return False

        metric_map = {m.metric: m for m in metrics}

        # WER threshold (raw value, lower is better)
        max_wer = t.get("max_wer", 0.15)
        wer_metric = metric_map.get("wer")
        if wer_metric and wer_metric.raw_value is not None:
            if wer_metric.raw_value > max_wer:
                return False

        # Latency threshold (raw ms)
        max_latency = t.get("max_latency_ms", 1000)
        lat_metric = metric_map.get("latency")
        if lat_metric and lat_metric.raw_value is not None:
            if lat_metric.raw_value > max_latency:
                return False

        # Semantic similarity threshold
        min_sem = t.get("min_semantic_similarity", 0.75)
        sem_metric = metric_map.get("semantic_similarity")
        if sem_metric and sem_metric.error is None:
            if sem_metric.score < min_sem:
                return False

        # Hallucination rate threshold (raw value)
        max_hall = t.get("max_hallucination_rate", 0.10)
        hall_metric = metric_map.get("hallucination")
        if hall_metric and hall_metric.raw_value is not None:
            if hall_metric.raw_value > max_hall:
                return False

        return True

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_evaluators(self) -> list[BaseEvaluator]:
        metrics_cfg = self.config.get("metrics", {})
        ollama_cfg = self.config.get("ollama", {})
        evaluators: list[BaseEvaluator] = []

        if metrics_cfg.get("wer", {}).get("enabled", True):
            evaluators.append(WERevaluator(config=metrics_cfg.get("wer", {})))

        if metrics_cfg.get("latency", {}).get("enabled", True):
            lat_cfg = {
                **metrics_cfg.get("latency", {}),
                "thresholds": self.config.get("scoring", {}).get(
                    "latency_thresholds", {}
                ),
            }
            evaluators.append(LatencyEvaluator(config=lat_cfg))

        if metrics_cfg.get("semantic_similarity", {}).get("enabled", True):
            sem_cfg = {
                **metrics_cfg.get("semantic_similarity", {}),
                "ollama": ollama_cfg,
                "seed": self.seed,
            }
            evaluators.append(SemanticEvaluator(config=sem_cfg))

        if metrics_cfg.get("hallucination", {}).get("enabled", True):
            hall_cfg = {
                **metrics_cfg.get("hallucination", {}),
                "ollama": ollama_cfg,
                "seed": self.seed,
            }
            evaluators.append(HallucinationEvaluator(config=hall_cfg))

        return evaluators

    def _build_transcriber(self, mock: bool):
        if mock:
            return MockTranscriber()
        w = self.config.get("whisper", {})
        return AudioTranscriber(
            model_size=w.get("model_size", "base"),
            device=w.get("device", "cpu"),
            language=w.get("language", "en"),
            fp16=w.get("fp16", False),
            decode_options=w.get("decode_options"),
        )

    def _build_client(self, mock: bool) -> BaseVoiceAIClient:
        if mock:
            return MockVoiceAIClient(fixed_latency_ms=150.0)
        # Default to Ollama client for standalone operation
        ollama_cfg = self.config.get("ollama", {})
        return OllamaVoiceAIClient(
            model=ollama_cfg.get("model", "llama3"),
            host=ollama_cfg.get("host", "http://localhost:11434"),
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _print_summary(self, summary) -> None:
        console.print()
        console.rule("[bold green]Evaluation Summary")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Cases", str(summary.total_cases))
        table.add_row(
            "Pass Rate",
            f"[green]{summary.pass_rate:.1%}[/green]"
            if summary.pass_rate >= 0.8
            else f"[red]{summary.pass_rate:.1%}[/red]",
        )
        if summary.avg_wer is not None:
            table.add_row("Avg WER", f"{summary.avg_wer:.4f}")
        if summary.avg_latency_ms is not None:
            table.add_row("Avg Latency", f"{summary.avg_latency_ms:.1f} ms")
        if summary.p95_latency_ms is not None:
            table.add_row("P95 Latency", f"{summary.p95_latency_ms:.1f} ms")
        if summary.avg_semantic_similarity is not None:
            table.add_row("Avg Semantic Sim", f"{summary.avg_semantic_similarity:.4f}")
        if summary.avg_hallucination_rate is not None:
            table.add_row("Avg Hallucination Rate", f"{summary.avg_hallucination_rate:.4f}")
        if summary.avg_composite_score is not None:
            table.add_row(
                "Avg Composite Score",
                f"[bold]{summary.avg_composite_score:.4f}[/bold]",
            )

        console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    configure_root_logger()

    parser = argparse.ArgumentParser(
        description="Voice AI Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml",
    )
    parser.add_argument(
        "--test-cases",
        help="Path to test cases JSON file",
    )
    parser.add_argument(
        "--audio-dir",
        help="Directory of audio files to evaluate",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock transcriber and Voice AI client (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from config",
    )

    args = parser.parse_args()

    if not args.test_cases and not args.audio_dir:
        parser.error("Provide --test-cases or --audio-dir")

    config = load_config(args.config)

    if args.seed is not None:
        config.setdefault("evaluation", {})["seed"] = args.seed

    if args.test_cases:
        test_cases = load_test_cases(args.test_cases)
    else:
        # Build test cases from audio directory
        audio_dir = Path(args.audio_dir)
        test_cases = [
            {
                "id": f"audio_{p.stem}",
                "category": "audio",
                "input_text": "",
                "audio_file": str(p),
                "expected_response": "",
                "reference_facts": [],
            }
            for p in sorted(audio_dir.glob("**/*.wav"))
            + sorted(audio_dir.glob("**/*.mp3"))
        ]

    orchestrator = EvaluationOrchestrator(config=config, mock=args.mock)
    report_path = orchestrator.run(test_cases)

    console.print(f"\n[bold green]✓ Report saved: {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
