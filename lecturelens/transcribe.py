"""Speech to text with pluggable backends.

faster-whisper : CPU baseline, easiest to start with.
whisper-cpp    : calls a whisper.cpp binary (good fit for Windows on ARM).
qnn            : integration point for an AI Hub Whisper model on the NPU (see docs/NPU_SETUP.md).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path


def _to_wav(path):
    if Path(path).suffix.lower() == ".wav":
        return str(path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert non-wav audio. Install it and retry.")
    out = Path(tempfile.mkdtemp()) / "audio.wav"
    subprocess.run([ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(out)],
                   check=True, capture_output=True)
    return str(out)


def _faster_whisper(audio, cfg):
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio), language=cfg.get("language"))
    return [s.text.strip() for s in segments]


def _whisper_cpp(audio, cfg):
    cmd = [cfg["whisper_cpp_bin"], "-m", cfg["whisper_cpp_model"], "-f", _to_wav(audio), "-nt"]
    if cfg.get("language"):
        cmd += ["-l", cfg["language"]]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _qnn(audio, cfg):
    raise NotImplementedError(
        "The NPU Whisper backend is an integration point. Export a Whisper model from "
        "Qualcomm AI Hub and wire it here. See docs/NPU_SETUP.md."
    )


BACKENDS = {"faster-whisper": _faster_whisper, "whisper-cpp": _whisper_cpp, "qnn": _qnn}


def transcribe(audio_path, cfg):
    """Return the transcript with one segment per line."""
    backend = cfg["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"Unknown transcribe.backend '{backend}'. Choose from {list(BACKENDS)}.")
    return "\n".join(BACKENDS[backend](audio_path, cfg))
