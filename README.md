# Voice AI Evaluation Pipeline

A rigorous, deterministic evaluation framework for Voice AI systems, measuring latency, Word Error Rate (WER), semantic similarity, and hallucination rate.

---

## Project Structure

```
voice_ai_eval/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
│
├── configs/
│   ├── eval_config.yaml          # Main evaluation configuration
│   └── ollama_config.yaml        # Ollama/LLM configuration
│
├── data/
│   └── samples/
│       ├── sample_test_cases.json # Example evaluation test cases
│       └── README.md
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py           # Main evaluation orchestrator
│   ├── audio_transcriber.py      # Whisper-based transcription pipeline
│   └── voice_ai_client.py        # Voice AI system interface
│
├── evaluators/
│   ├── __init__.py
│   ├── base_evaluator.py         # Abstract base evaluator
│   ├── wer_evaluator.py          # Word Error Rate evaluator
│   ├── latency_evaluator.py      # Latency measurement evaluator
│   ├── semantic_evaluator.py     # Semantic similarity (Ollama-powered)
│   └── hallucination_evaluator.py# Hallucination detection (Ollama-powered)
│
├── utils/
│   ├── __init__.py
│   ├── determinism.py            # Determinism guarantees & seeding
│   ├── report_generator.py       # JSON report generation
│   └── logger.py                 # Structured logging
│
├── tests/
│   ├── __init__.py
│   ├── test_wer_evaluator.py
│   ├── test_latency_evaluator.py
│   ├── test_semantic_evaluator.py
│   ├── test_hallucination_evaluator.py
│   ├── test_audio_transcriber.py
│   ├── test_orchestrator.py
│   └── test_report_generator.py
│
└── reports/                      # Generated evaluation reports (git-ignored)
    └── .gitkeep
```

---

## Quickstart

### 1. Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running locally
- ffmpeg (for audio processing)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama3)
ollama pull llama3

# Start Ollama server
ollama serve
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run evaluation

```bash
# Run with sample test cases
python -m pipeline.orchestrator --config configs/eval_config.yaml --test-cases data/samples/sample_test_cases.json

# Run with audio files
python -m pipeline.orchestrator --config configs/eval_config.yaml --audio-dir /path/to/audio/

# Run unit tests
pytest tests/ -v --tb=short
```

---

## Architecture

### Determinism Guarantees

All evaluation runs are deterministic via:
- Fixed random seeds (configurable, default: `42`)
- Frozen LLM parameters (`temperature=0`, `top_p=1`, `seed` fixed)
- Canonical text normalization before WER/similarity scoring
- SHA-256 hashing of inputs logged in every report for audit

### Metrics

| Metric | Method | Range |
|--------|--------|-------|
| **Latency** | Wall-clock + TTFB measurement | ms |
| **WER** | Levenshtein edit distance on normalized tokens | 0–1 (lower=better) |
| **Semantic Similarity** | LLM judge + sentence-transformers cosine | 0–1 (higher=better) |
| **Hallucination Rate** | LLM-based factual grounding check | 0–1 (lower=better) |

### Report Format

Reports are written to `reports/` as timestamped JSON files:

```json
{
  "run_id": "eval_20240315_143022",
  "config_hash": "sha256:abc123...",
  "seed": 42,
  "timestamp": "2024-03-15T14:30:22Z",
  "summary": {
    "total_cases": 10,
    "avg_wer": 0.08,
    "avg_latency_ms": 342.5,
    "avg_semantic_similarity": 0.91,
    "avg_hallucination_rate": 0.05
  },
  "cases": [...]
}
```

---

## Configuration

Edit `configs/eval_config.yaml` to customize:
- Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- Ollama model and endpoint
- Scoring thresholds and weights
- Output paths

---

## Extending the Framework

To add a new evaluator:

1. Create `evaluators/my_evaluator.py` extending `BaseEvaluator`
2. Implement `evaluate(test_case, response)` returning an `EvalResult`
3. Register it in `pipeline/orchestrator.py`

```python
from evaluators.base_evaluator import BaseEvaluator, EvalResult

class MyEvaluator(BaseEvaluator):
    name = "my_metric"

    def evaluate(self, test_case, response) -> EvalResult:
        score = compute_my_score(test_case, response)
        return EvalResult(metric=self.name, score=score, details={})
```
