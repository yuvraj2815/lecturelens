"""Download the open-source embedding model (ONNX) used for textbook search."""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "sentence-transformers/all-MiniLM-L6-v2"
DEST = Path("models/minilm")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for remote, local in [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
        shutil.copy(hf_hub_download(REPO, remote), DEST / local)
        print(f"Saved {DEST / local}")


if __name__ == "__main__":
    main()
