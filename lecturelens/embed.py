"""Sentence embeddings (MiniLM-class ONNX model) used for textbook search."""
import numpy as np

from .runtime import make_session


class Embedder:
    def __init__(self, model_path, tokenizer_path, max_len=128, prefer="qnn",
                 backend_path="QnnHtp.dll"):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(length=max_len)  # static shapes suit NPU compilation
        self.session = make_session(model_path, prefer, backend_path)
        self.input_names = {i.name for i in self.session.get_inputs()}

    @property
    def providers(self):
        return self.session.get_providers()

    def _encode_one(self, text):
        enc = self.tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        feed = {k: v for k, v in feed.items() if k in self.input_names}
        hidden = self.session.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return vec[0] / max(float(np.linalg.norm(vec[0])), 1e-9)

    def embed(self, texts):
        return np.stack([self._encode_one(t) for t in texts]).astype(np.float32)
