from lecturelens.config import load_config


def test_defaults_load_without_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg["runtime"]["prefer"] == "qnn"
    assert cfg["index"]["top_k"] == 4


def test_override_merges_nested(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("index:\n  top_k: 9\nllm:\n  model: phi3.5\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg["index"]["top_k"] == 9
    assert cfg["index"]["chunk_chars"] == 900  # untouched sibling key survives
    assert cfg["llm"]["model"] == "phi3.5"
