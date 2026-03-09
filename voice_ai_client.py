"""
Voice AI Client — interface to the system under test.

This module abstracts the Voice AI system behind a clean interface.
Swap out the implementation to test any system:
- HTTP REST API (default)
- Local Python function
- Mock/stub for testing

Latency measurement is done here using perf_counter for sub-millisecond
precision, capturing both Time-to-First-Byte (TTFB) and total latency.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VoiceAIResponse:
    """Structured response from the Voice AI system."""

    text: str
    latency_ms: float
    ttfb_ms: float | None = None
    status_code: int | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


class BaseVoiceAIClient(ABC):
    """Abstract client for the Voice AI system under test."""

    @abstractmethod
    def query(self, input_text: str, **kwargs) -> VoiceAIResponse:
        raise NotImplementedError

    def query_with_audio(
        self, audio_path: str, transcribed_text: str, **kwargs
    ) -> VoiceAIResponse:
        """Default: use the transcription as text input."""
        return self.query(transcribed_text, **kwargs)


class HTTPVoiceAIClient(BaseVoiceAIClient):
    """
    HTTP REST client for a Voice AI API.

    Expected API contract:
    POST /query
    Body: {"text": "...", "session_id": "..."}
    Response: {"response": "...", "metadata": {...}}
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def query(self, input_text: str, **kwargs) -> VoiceAIResponse:
        """Send a text query and return the response with timing."""
        url = f"{self.endpoint}/query"
        payload = {"text": input_text, **kwargs}

        ttfb_ms = None
        t_start = time.perf_counter()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload, headers=self.headers) as resp:
                    # Record TTFB at first byte
                    ttfb_ms = (time.perf_counter() - t_start) * 1000
                    resp.read()  # Read full response

            t_total = (time.perf_counter() - t_start) * 1000

            if resp.status_code != 200:
                return VoiceAIResponse(
                    text="",
                    latency_ms=round(t_total, 2),
                    ttfb_ms=round(ttfb_ms, 2),
                    status_code=resp.status_code,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()
            response_text = data.get("response", data.get("text", ""))

            return VoiceAIResponse(
                text=response_text,
                latency_ms=round(t_total, 2),
                ttfb_ms=round(ttfb_ms, 2),
                status_code=resp.status_code,
                raw_response=data,
            )

        except Exception as exc:
            t_total = (time.perf_counter() - t_start) * 1000
            logger.error(f"Voice AI query failed: {exc}")
            return VoiceAIResponse(
                text="",
                latency_ms=round(t_total, 2),
                ttfb_ms=ttfb_ms,
                error=f"{type(exc).__name__}: {exc}",
            )


class OllamaVoiceAIClient(BaseVoiceAIClient):
    """
    Use Ollama directly as the Voice AI system under test.
    Useful for evaluating LLM response quality directly.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        seed: int = 42,
    ):
        self.model = model
        self.host = host
        self.seed = seed
        # Fixed options for deterministic evaluation
        from utils.determinism import get_fixed_ollama_options
        self.options = get_fixed_ollama_options(seed)

    def query(self, input_text: str, **kwargs) -> VoiceAIResponse:
        import ollama

        t_start = time.perf_counter()
        ttfb_ms = None

        try:
            # Use streaming to capture TTFB
            response_parts = []
            for chunk in ollama.generate(
                model=self.model,
                prompt=input_text,
                options=self.options,
                host=self.host,
                stream=True,
            ):
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - t_start) * 1000
                response_parts.append(chunk.get("response", ""))

            t_total = (time.perf_counter() - t_start) * 1000
            full_response = "".join(response_parts).strip()

            return VoiceAIResponse(
                text=full_response,
                latency_ms=round(t_total, 2),
                ttfb_ms=round(ttfb_ms, 2) if ttfb_ms else None,
            )

        except Exception as exc:
            t_total = (time.perf_counter() - t_start) * 1000
            return VoiceAIResponse(
                text="",
                latency_ms=round(t_total, 2),
                error=f"{type(exc).__name__}: {exc}",
            )


class MockVoiceAIClient(BaseVoiceAIClient):
    """
    Mock client for unit testing.
    Returns the expected_response from the test case if provided,
    otherwise returns a fixed stub response.
    """

    def __init__(self, fixed_latency_ms: float = 150.0):
        self.fixed_latency_ms = fixed_latency_ms

    def query(self, input_text: str, **kwargs) -> VoiceAIResponse:
        # Simulate deterministic latency
        expected = kwargs.get("_expected_response", f"Mock response to: {input_text}")
        return VoiceAIResponse(
            text=expected,
            latency_ms=self.fixed_latency_ms,
            ttfb_ms=self.fixed_latency_ms * 0.3,
        )
