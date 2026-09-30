"""Prompt templates. Notes are exam-oriented: topic-wise, with formulas called out."""

NOTES_SYSTEM = (
    "You are a careful study assistant. Explain concepts from the basics, then give exam-ready "
    "points. Never invent facts that are not in the source text."
)

NOTES_USER = """Turn this lecture transcript on {subject} into study notes in Markdown.

Structure:
## <Topic name>
- Concept in plain words (2 to 3 lines)
- **Key formulas:** each formula on its own line, with what every symbol means
- **Exam points:** short bullets likely to be asked

If the transcript is unclear, say so instead of guessing.

Transcript:
{transcript}
"""

MERGE_USER = """Merge these partial study notes into one clean set of notes.
Remove duplicates, group by topic, and keep every formula.

{notes}
"""

QUIZ_USER = """Write {n} multiple-choice questions from these notes.
For each: the question, four options (A to D), the correct answer, and a one-line explanation.

Notes:
{notes}
"""

QA_SYSTEM = (
    "Answer only from the provided textbook excerpts. Cite the page number like (p. 42) after "
    "each claim. If the excerpts do not contain the answer, say you could not find it."
)

QA_USER = """Excerpts:
{context}

Question: {question}
"""
