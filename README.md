# LectureLens

A private, offline AI study copilot for Snapdragon-powered HP PCs. It turns lectures and textbooks into exam-ready notes, formulas, cited answers and quizzes. Nothing leaves the laptop.

![CI](https://github.com/YOUR-USERNAME/lecturelens/actions/workflows/ci.yml/badge.svg)

Built for the Snapdragon AI Lab Challenge. Pitch deck: [`docs/pitch/`](docs/pitch/).

## Features

- **Live lecture mode:** transcribe audio, then generate topic-wise notes with key formulas and a quiz.
- **Book study mode:** index a textbook PDF and ask questions. Answers cite page numbers.
- **Offline by design:** the only network calls go to a model server on `localhost`.
- **NPU-ready runtime:** ONNX Runtime with the Qualcomm QNN provider, with a CPU fallback.
- **Hinglish-friendly:** speech backends accept a language hint (for example `en` or `hi`).

## Status

This is a working prototype scaffold. Be clear about what is and is not done:

| Component | State |
|---|---|
| Textbook indexing, retrieval, cited Q&A | Implemented |
| Notes and quiz generation (map then merge) | Implemented, needs a local LLM server |
| Embeddings via ONNX Runtime with QNN provider | Implemented, benchmark included |
| Speech to text | CPU baselines implemented; NPU backend is an integration point |
| LLM on the NPU | Integration point, see [`docs/NPU_SETUP.md`](docs/NPU_SETUP.md) |

Measured results belong in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Quick start (Windows on Snapdragon)

```powershell
git clone https://github.com/YOUR-USERNAME/lecturelens.git
cd lecturelens
.\scripts\setup_windows.ps1
```

Or manually, with a native ARM64 Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-optional.txt   # faster-whisper, if a wheel exists for your platform
python scripts/download_models.py
python -m lecturelens providers            # look for QNNExecutionProvider
```

Start a local model server that speaks the OpenAI chat API (for example Ollama with `llama3.2:3b`), then edit `llm` in `config.yaml` if needed.

## Usage

```powershell
python app.py                                             # web UI

python -m lecturelens transcribe lecture.mp3
python -m lecturelens notes outputs/transcript.txt -s "Thermodynamics"
python -m lecturelens quiz outputs/notes.md -n 5

python -m lecturelens index textbook.pdf --name thermo
python -m lecturelens ask --name thermo "State the first law of thermodynamics"

python -m lecturelens bench                               # CPU vs NPU embedding latency
```

## Models

| Task | Model class | Source |
|---|---|---|
| Speech to text | Whisper (base or small) | Qualcomm AI Hub or open source |
| Notes and Q&A | Llama 3.2 3B or Phi-3.5 mini, quantized | Qualcomm AI Hub or open source |
| Semantic search | all-MiniLM-L6-v2 (ONNX) | Hugging Face |

## Project layout

```
lecturelens/   core package (runtime, transcribe, embed, library, llm, notes, bench, cli)
app.py         Gradio web UI
config.yaml    settings
scripts/       model download and Windows setup
docs/          architecture, NPU setup, benchmarks, pitch deck
tests/         unit tests (run with pytest)
```

## Development

```powershell
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Roadmap

1. Phase 1: offline transcription on the NPU and topic-wise note generation.
2. Phase 2: textbook indexing with cited Q&A, quizzes and flashcards, Hinglish tuning.
3. Phase 3: more Indian languages, subject packs and formula sheets, campus pilots.

## License

MIT. See [`LICENSE`](LICENSE).
