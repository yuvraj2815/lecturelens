import numpy as np

from lecturelens.library import chunk_pages, search


def test_chunks_keep_page_numbers_and_overlap():
    pages = [(1, "a" * 2000), (2, "short page")]
    chunks = chunk_pages(pages, chunk_chars=900, overlap_chars=150)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(len(c["text"]) <= 900 for c in chunks)
    assert len([c for c in chunks if c["page"] == 1]) == 3


def test_empty_pages_are_skipped():
    assert chunk_pages([(1, "   "), (2, "")]) == []


def test_search_returns_best_match_first():
    vecs = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    chunks = [{"page": 1, "text": "x"}, {"page": 2, "text": "y"}, {"page": 3, "text": "z"}]
    hits = search(np.array([0, 1], dtype=np.float32), vecs, chunks, k=2)
    assert hits[0]["page"] == 2
    assert len(hits) == 2
