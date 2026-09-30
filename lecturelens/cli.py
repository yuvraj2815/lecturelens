"""Command line interface: python -m lecturelens <command>."""
import argparse
from pathlib import Path

from .config import load_config


def _embedder(cfg):
    from .embed import Embedder

    e, rt = cfg["embedding"], cfg["runtime"]
    return Embedder(e["model_path"], e["tokenizer_path"], e["max_len"], rt["prefer"],
                    rt["backend_path"])


def _write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    print(f"Saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lecturelens", description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("providers", help="show ONNX Runtime execution providers")
    t = sub.add_parser("transcribe", help="audio -> transcript")
    t.add_argument("audio"); t.add_argument("-o", "--out", default="outputs/transcript.txt")
    n = sub.add_parser("notes", help="transcript -> exam notes")
    n.add_argument("transcript"); n.add_argument("-s", "--subject", default="")
    n.add_argument("-o", "--out", default="outputs/notes.md")
    q = sub.add_parser("quiz", help="notes -> MCQ quiz")
    q.add_argument("notes"); q.add_argument("-n", type=int, default=5)
    q.add_argument("-o", "--out", default="outputs/quiz.md")
    i = sub.add_parser("index", help="index a textbook PDF")
    i.add_argument("pdf"); i.add_argument("--name", required=True)
    a = sub.add_parser("ask", help="cited question answering over an indexed book")
    a.add_argument("--name", required=True); a.add_argument("question")
    b = sub.add_parser("bench", help="embedding latency, CPU vs NPU")
    b.add_argument("-n", type=int, default=50)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "providers":
        from .runtime import available_providers
        print("\n".join(available_providers()))
    elif args.cmd == "transcribe":
        from .transcribe import transcribe
        _write(args.out, transcribe(args.audio, cfg["transcribe"]))
    elif args.cmd == "notes":
        from .llm import LLM
        from .notes import generate_notes
        text = Path(args.transcript).read_text(encoding="utf-8")
        _write(args.out, generate_notes(text, LLM.from_config(cfg), args.subject))
    elif args.cmd == "quiz":
        from .llm import LLM
        from .notes import generate_quiz
        text = Path(args.notes).read_text(encoding="utf-8")
        _write(args.out, generate_quiz(text, LLM.from_config(cfg), args.n))
    elif args.cmd == "index":
        from .library import build_index
        ix = cfg["index"]
        count = build_index(args.name, args.pdf, _embedder(cfg), ix["dir"], ix["chunk_chars"],
                            ix["overlap_chars"])
        print(f"Indexed {count} chunks as '{args.name}'")
    elif args.cmd == "ask":
        from .library import load_index, search
        from .llm import LLM
        from .notes import answer_question
        ix = cfg["index"]
        vecs, chunks = load_index(args.name, ix["dir"])
        qv = _embedder(cfg).embed([args.question])[0]
        hits = search(qv, vecs, chunks, ix["top_k"])
        print(answer_question(args.question, hits, LLM.from_config(cfg)))
    elif args.cmd == "bench":
        from .bench import run, to_markdown
        print(to_markdown(run(cfg, args.n)))
