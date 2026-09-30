"""Configuration loading: built-in defaults, overridden by config.yaml."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "runtime": {"prefer": "qnn", "backend_path": "QnnHtp.dll"},
    "transcribe": {
        "backend": "faster-whisper",  # faster-whisper | whisper-cpp | qnn
        "model": "base",
        "language": None,
        "whisper_cpp_bin": "whisper-cli",
        "whisper_cpp_model": "models/ggml-base.bin",
    },
    "embedding": {
        "model_path": "models/minilm/model.onnx",
        "tokenizer_path": "models/minilm/tokenizer.json",
        "max_len": 128,
    },
    "llm": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
        "temperature": 0.2,
        "max_tokens": 1200,
    },
    "index": {"dir": "data/index", "chunk_chars": 900, "overlap_chars": 150, "top_k": 4},
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path or "config.yaml")
    if path.exists():
        _merge(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return cfg
