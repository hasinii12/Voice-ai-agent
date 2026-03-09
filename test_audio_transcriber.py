"""
Tests for AudioTranscriber.

Real Whisper calls are mocked so no audio files or GPU needed in CI.
Tests verify: deterministic options, result structure, error handling,
file validation, and batch transcription.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pipeline.audio_transcriber import (
    AudioTranscriber,
    MockTranscriber,
    TranscriptionResult,
    SUPPORTED_FORMATS,
)
from utils.determinism import seed_everything, get_fixed_whisper_options


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def mock_whisper_model():
    """Mock whisper model that returns a fixed transcription."""
    model = MagicMock()
    model.transcribe.return_value = {
        "text": " The capital of France is Paris.",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": " The capital of France is Paris."}
        ],
    }
    return model


@pytest.fixture
def tmp_audio_file(tmp_path):
    """Create a minimal temp WAV file for path validation tests."""
    f = tmp_path / "test.wav"
    f.write_bytes(b"RIFF" + b"\x00" * 100)  # Fake WAV content
    return f


class TestAudioTranscriberConfig:

    def test_deterministic_options_set(self):
        transcriber = AudioTranscriber(model_size="base", device="cpu")
        fixed_opts = get_fixed_whisper_options()
        assert transcriber.decode_options["temperature"] == 0.0
        assert transcriber.decode_options["beam_size"] == fixed_opts["beam_size"]
        assert transcriber.decode_options["condition_on_previous_text"] is False

    def test_custom_options_merged(self):
        transcriber = AudioTranscriber(decode_options={"beam_size": 3})
        # Custom override applied
        assert transcriber.decode_options["beam_size"] == 3
        # But deterministic defaults still present
        assert transcriber.decode_options["temperature"] == 0.0

    def test_fp16_flag_applied(self):
        transcriber = AudioTranscriber(fp16=True)
        assert transcriber.decode_options["fp16"] is True


class TestAudioTranscriberTranscribe:

    @patch("pipeline.audio_transcriber.whisper")
    def test_transcription_returns_correct_text(
        self, mock_whisper, mock_whisper_model, tmp_audio_file
    ):
        mock_whisper.load_model.return_value = mock_whisper_model
        transcriber = AudioTranscriber()
        result = transcriber.transcribe(tmp_audio_file)

        assert isinstance(result, TranscriptionResult)
        assert "Paris" in result.text
        assert result.language == "en"

    @patch("pipeline.audio_transcriber.whisper")
    def test_transcription_metadata(
        self, mock_whisper, mock_whisper_model, tmp_audio_file
    ):
        mock_whisper.load_model.return_value = mock_whisper_model
        transcriber = AudioTranscriber(model_size="small")
        result = transcriber.transcribe(tmp_audio_file)

        assert result.model_name == "whisper-small"
        assert result.audio_hash != ""
        assert result.transcription_time_ms >= 0
        assert result.duration_seconds == pytest.approx(2.5)

    @patch("pipeline.audio_transcriber.whisper")
    def test_deterministic_decode_options_passed_to_model(
        self, mock_whisper, mock_whisper_model, tmp_audio_file
    ):
        mock_whisper.load_model.return_value = mock_whisper_model
        transcriber = AudioTranscriber()
        transcriber.transcribe(tmp_audio_file)

        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["condition_on_previous_text"] is False

    @patch("pipeline.audio_transcriber.whisper")
    def test_same_file_same_result(
        self, mock_whisper, mock_whisper_model, tmp_audio_file
    ):
        mock_whisper.load_model.return_value = mock_whisper_model
        transcriber = AudioTranscriber()
        r1 = transcriber.transcribe(tmp_audio_file)
        r2 = transcriber.transcribe(tmp_audio_file)
        assert r1.text == r2.text
        assert r1.audio_hash == r2.audio_hash


class TestAudioTranscriberValidation:

    def test_missing_file_raises(self):
        transcriber = AudioTranscriber.__new__(AudioTranscriber)
        transcriber.decode_options = {}
        with pytest.raises(FileNotFoundError):
            transcriber._validate_audio_file(Path("/nonexistent/file.wav"))

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "audio.xyz"
        f.write_bytes(b"data")
        transcriber = AudioTranscriber.__new__(AudioTranscriber)
        with pytest.raises(ValueError, match="Unsupported audio format"):
            transcriber._validate_audio_file(f)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        transcriber = AudioTranscriber.__new__(AudioTranscriber)
        with pytest.raises(ValueError, match="empty"):
            transcriber._validate_audio_file(f)

    def test_all_supported_formats_pass(self, tmp_path):
        transcriber = AudioTranscriber.__new__(AudioTranscriber)
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"audio{fmt}"
            f.write_bytes(b"data")
            transcriber._validate_audio_file(f)  # Should not raise


class TestMockTranscriber:

    def test_mock_returns_transcription_result(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"data")
        transcriber = MockTranscriber()
        result = transcriber.transcribe(f)
        assert isinstance(result, TranscriptionResult)
        assert result.model_name == "mock"
        assert "MOCK" in result.text

    def test_mock_batch_transcription(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"audio_{i}.wav"
            f.write_bytes(b"data")
            files.append(f)

        transcriber = MockTranscriber()
        results = transcriber.transcribe_batch(files)
        assert len(results) == 3
        assert all(isinstance(r, TranscriptionResult) for r in results)
