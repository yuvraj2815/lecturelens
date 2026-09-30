"""Client for a local OpenAI-compatible chat server (Ollama, llama.cpp, LM Studio)."""
import requests


class LLM:
    def __init__(self, base_url, model, temperature=0.2, max_tokens=1200, timeout=600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, system, user):
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach the local LLM at {self.url}. Start your local model server "
                f"and check llm.base_url in config.yaml. ({exc})"
            ) from exc
        return resp.json()["choices"][0]["message"]["content"].strip()

    @classmethod
    def from_config(cls, cfg):
        c = cfg["llm"]
        return cls(c["base_url"], c["model"], c["temperature"], c["max_tokens"])
