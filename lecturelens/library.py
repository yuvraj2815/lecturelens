"""Textbook ingestion and retrieval: PDF -> page chunks -> vectors -> top-k search."""
import json
from pathlib import Path

import numpy as np


def extract_pages(pdf_path):
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]


def chunk_pages(pages, chunk_chars=900, overlap_chars=150):
    """Split each page into overlapping chunks, keeping the page number for citations."""
    step = max(chunk_chars - overlap_chars, 1)
    chunks = []
    for page, text in pages:
        text = " ".join(text.split())
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_chars]
            if piece.strip():
                chunks.append({"page": page, "text": piece})
            if start + chunk_chars >= len(text):
                break
    return chunks


def build_index(name, pdf_path, embedder, out_dir, chunk_chars=900, overlap_chars=150):
    chunks = chunk_pages(extract_pages(pdf_path), chunk_chars, overlap_chars)
    if not chunks:
        raise ValueError("No extractable text found. Scanned PDFs need an OCR step first.")
    vecs = embedder.embed([c["text"] for c in chunks])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.npy", vecs)
    (out / f"{name}.json").write_text(json.dumps(chunks), encoding="utf-8")
    return len(chunks)


def load_index(name, out_dir):
    out = Path(out_dir)
    vecs = np.load(out / f"{name}.npy")
    chunks = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    return vecs, chunks


def search(query_vec, vecs, chunks, k=4):
    scores = vecs @ query_vec
    top = np.argsort(-scores)[:k]
    return [dict(chunks[i], score=float(scores[i])) for i in top]
