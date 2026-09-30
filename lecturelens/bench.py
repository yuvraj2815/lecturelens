"""Latency benchmark: the same embedding model on CPU versus the NPU (QNN) provider."""
import statistics
import time

from .embed import Embedder

SENTENCES = [
    "The first law of thermodynamics relates heat, work and internal energy.",
    "Bernoulli's equation links pressure, velocity and elevation along a streamline.",
    "A primary key uniquely identifies each row in a relational table.",
    "Gradient descent updates parameters in the direction of the negative gradient.",
    "The Reynolds number predicts whether a flow is laminar or turbulent.",
]


def run(cfg, n=50):
    e, rt = cfg["embedding"], cfg["runtime"]
    rows = []
    for requested in ("cpu", "qnn"):
        emb = Embedder(e["model_path"], e["tokenizer_path"], e["max_len"], requested,
                       rt["backend_path"])
        emb.embed(SENTENCES[:2])  # warm-up
        times = []
        for i in range(n):
            start = time.perf_counter()
            emb.embed([SENTENCES[i % len(SENTENCES)]])
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows.append((requested, emb.providers[0], statistics.median(times),
                     times[int(0.95 * (len(times) - 1))]))
    return rows


def to_markdown(rows):
    lines = ["| Requested | Active provider | Median (ms) | p95 (ms) |", "|---|---|---|---|"]
    lines += [f"| {r} | {a} | {m:.2f} | {p:.2f} |" for r, a, m, p in rows]
    return "\n".join(lines)
