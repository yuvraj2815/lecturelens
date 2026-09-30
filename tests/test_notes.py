from lecturelens.notes import generate_notes, split_text


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        return f"notes-{self.calls}"


def test_split_text_respects_limit():
    text = "\n".join(["line " * 20] * 200)
    parts = split_text(text, max_chars=1000)
    assert len(parts) > 1
    assert all(len(p) <= 1100 for p in parts)


def test_short_transcript_uses_single_call():
    llm = FakeLLM()
    assert generate_notes("hello\nworld", llm) == "notes-1"
    assert llm.calls == 1


def test_long_transcript_maps_then_merges():
    llm = FakeLLM()
    generate_notes("\n".join(["word " * 50] * 400), llm)
    assert llm.calls >= 3  # at least two parts plus one merge call
