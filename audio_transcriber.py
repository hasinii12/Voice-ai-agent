"""
Audio Transcription Pipeline using OpenAI Whisper.

Provides deterministic transcription of audio files to text.
Determinism is guaranteed by:
- Fixed model weights (Whisper base/small/medium)
- temperature=0 (greedy decoding)
- Fixed beam_size, best_of, and compression thresholds
- Disabled previous text conditioning to prevent cascading drift

Supports:
- WAV, MP3, FLAC, M4A, OGG, WEBM formats
- GPU/CPU/MPS device selection
- Optional file hash verification before transcription
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.determinism import get_fixed_whisper_options, hash_file
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".mp4"}


@dataclass
class TranscriptionResult:
    """Result from a Whisper transcription."""

    text: str
    language: str
    duration_seconds: float
    segments: list[dict[str, Any]]
    audio_file: str
    audio_hash: str
    transcription_time_ms: float
    model_name: str
    decode_options: dict[str, Any]


class AudioTranscriber:
    """
    Whisper-based audio transcription with determinism guarantees.

    Parameters
    ----------
    model_size   : Whisper model size (tiny/base/small/medium/large)
    device       : "cpu", "cuda", or "mps"
    language     : Language code (e.g. "en") or None for auto-detect
    fp16         : Use FP16 — only for GPU, introduces minor non-determinism
    decode_options: Override default decode options (merged, not replaced)
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "en",
        fp16: bool = False,
        decode_options: dict[str, Any] | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.fp16 = fp16

        # Merge user options with deterministic defaults
        base_opts = get_fixed_whisper_options()
        base_opts["fp16"] = fp16
        self.decode_options = {**base_opts, **(decode_options or {})}

        self._model = None  # Lazy-loaded
        logger.info(
            f"AudioTranscriber configured: model={model_size} device={device} "
            f"language={language} fp16={fp16}"
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Returns a TranscriptionResult with the transcript, metadata,
        and timing information.
        """
        audio_path = Path(audio_path)
        self._validate_audio_file(audio_path)

        model = self._get_model()
        audio_hash = hash_file(audio_path)

        logger.debug(f"Transcribing: {audio_path.name} (sha256:{audio_hash[:12]}...)")

        t0 = time.perf_counter()
        result = model.transcribe(
            str(audio_path),
            language=self.language,
            **self.decode_options,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        transcript = result.get("text", "").strip()
        language = result.get("language", self.language or "unknown")
        segments = result.get("segments", [])

        # Compute audio duration from segments
        duration = segments[-1]["end"] if segments else 0.0

        logger.info(
            f"Transcribed {audio_path.name} → {len(transcript)} chars "
            f"in {elapsed_ms:.0f}ms (RTF={elapsed_ms / (duration * 1000):.2f}x)"
            if duration > 0
            else f"Transcribed {audio_path.name} → {len(transcript)} chars in {elapsed_ms:.0f}ms"
        )

        return TranscriptionResult(
            text=transcript,
            language=language,
            duration_seconds=duration,
            segments=[
                {
                    "id": s.get("id"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text", "").strip(),
                }
                for s in segments
            ],
            audio_file=str(audio_path),
            audio_hash=audio_hash,
            transcription_time_ms=round(elapsed_ms, 2),
            model_name=f"whisper-{self.model_size}",
            decode_options=self.decode_options,
        )

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files sequentially.
        Returns results in the same order as input paths.
        """
        results = []
        for i, path in enumerate(audio_paths, 1):
            logger.info(f"Transcribing [{i}/{len(audio_paths)}]: {Path(path).name}")
            try:
                result = self.transcribe(path)
                results.append(result)
            except Exception as exc:
                logger.error(f"Transcription failed for {path}: {exc}")
                results.append(
                    TranscriptionResult(
                        text="",
                        language="unknown",
                        duration_seconds=0.0,
                        segments=[],
                        audio_file=str(path),
                        audio_hash="error",
                        transcription_time_ms=0.0,
                        model_name=f"whisper-{self.model_size}",
                        decode_options=self.decode_options,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load Whisper model to avoid slow startup on import."""
        if self._model is None:
            try:
                import whisper

                logger.info(f"Loading Whisper model: {self.model_size}")
                self._model = whisper.load_model(
                    self.model_size, device=self.device
                )
                logger.info(f"Whisper {self.model_size} loaded on {self.device}")
            except ImportError:
                raise ImportError(
                    "openai-whisper is not installed. "
                    "Run: pip install openai-whisper"
                )
        return self._model

    @staticmethod
    def _validate_audio_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {path}")


class MockTranscriber:
    """
    Mock transcriber for testing — returns the input_text directly.
    Useful for unit tests that don't need real audio files.
    """

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        return TranscriptionResult(
            text=f"[MOCK TRANSCRIPTION of {Path(audio_path).name}]",
            language="en",
            duration_seconds=1.0,
            segments=[],
            audio_file=str(audio_path),
            audio_hash="mock",
            transcription_time_ms=0.0,
            model_name="mock",
            decode_options={},
        )

    def transcribe_batch(self, paths):
        return [self.transcribe(p) for p in paths]
