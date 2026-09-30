"""Simple local web UI: python app.py  (opens on http://127.0.0.1:7860)."""
import gradio as gr

from lecturelens.config import load_config
from lecturelens.llm import LLM
from lecturelens.notes import answer_question, generate_notes, generate_quiz
from lecturelens.transcribe import transcribe

CFG = load_config()
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from lecturelens.embed import Embedder
        e, rt = CFG["embedding"], CFG["runtime"]
        _embedder = Embedder(e["model_path"], e["tokenizer_path"], e["max_len"], rt["prefer"],
                             rt["backend_path"])
    return _embedder


def lecture(audio, subject):
    if not audio:
        return "Upload or record audio first.", "", ""
    llm = LLM.from_config(CFG)
    transcript = transcribe(audio, CFG["transcribe"])
    notes = generate_notes(transcript, llm, subject)
    return transcript, notes, generate_quiz(notes, llm)


def index_book(pdf, name):
    from lecturelens.library import build_index
    ix = CFG["index"]
    count = build_index(name, pdf, get_embedder(), ix["dir"], ix["chunk_chars"],
                        ix["overlap_chars"])
    return f"Indexed {count} chunks as '{name}'."


def ask(name, question):
    from lecturelens.library import load_index, search
    ix = CFG["index"]
    vecs, chunks = load_index(name, ix["dir"])
    hits = search(get_embedder().embed([question])[0], vecs, chunks, ix["top_k"])
    return answer_question(question, hits, LLM.from_config(CFG))


with gr.Blocks(title="LectureLens") as demo:
    gr.Markdown("# LectureLens\nPrivate, offline study copilot. Nothing leaves this laptop.")
    with gr.Tab("Live lecture"):
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath")
        subject = gr.Textbox(label="Subject (optional)")
        go = gr.Button("Generate notes and quiz")
        tr, nt, qz = gr.Textbox(label="Transcript"), gr.Markdown(), gr.Markdown()
        go.click(lecture, [audio, subject], [tr, nt, qz])
    with gr.Tab("Book study"):
        pdf, bname = gr.File(type="filepath", label="Textbook PDF"), gr.Textbox(label="Book name")
        status = gr.Textbox(label="Status")
        gr.Button("Index book").click(index_book, [pdf, bname], status)
        question, answer = gr.Textbox(label="Question"), gr.Markdown()
        gr.Button("Ask").click(ask, [bname, question], answer)

if __name__ == "__main__":
    demo.launch()
