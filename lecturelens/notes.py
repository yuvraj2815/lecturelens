"""Notes, quiz and cited-answer generation on top of the local LLM."""
from . import prompts


def split_text(text, max_chars=6000):
    """Split a transcript on line boundaries so each part fits the model context."""
    parts, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) > max_chars:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts


def generate_notes(transcript, llm, subject=""):
    subject = subject or "the lecture"
    partial = [
        llm.chat(prompts.NOTES_SYSTEM, prompts.NOTES_USER.format(subject=subject, transcript=part))
        for part in split_text(transcript)
    ]
    if len(partial) == 1:
        return partial[0]
    return llm.chat(prompts.NOTES_SYSTEM, prompts.MERGE_USER.format(notes="\n\n".join(partial)))


def generate_quiz(notes, llm, n=5):
    return llm.chat(prompts.NOTES_SYSTEM, prompts.QUIZ_USER.format(n=n, notes=notes))


def answer_question(question, hits, llm):
    context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
    return llm.chat(prompts.QA_SYSTEM, prompts.QA_USER.format(context=context, question=question))
