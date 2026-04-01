import json
import os

import requests


class LLMRouter:
    """Minimal local-only router for Ollama-backed text generation."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")

    def call_local_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        request_timeout_s: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            "Return only the final answer.\n\n"
            f"{user_prompt.strip()}"
        )
        try:
            resolved_timeout = float(
                request_timeout_s
                if request_timeout_s is not None
                else os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "120")
            )
        except (TypeError, ValueError):
            resolved_timeout = 120.0
        if resolved_timeout <= 0:
            resolved_timeout = 120.0
        options = {"temperature": temperature}
        if num_predict is not None:
            try:
                resolved_num_predict = int(num_predict)
            except (TypeError, ValueError):
                resolved_num_predict = None
            if resolved_num_predict is not None and resolved_num_predict > 0:
                options["num_predict"] = resolved_num_predict
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options,
                },
                timeout=resolved_timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self._base_url}: {exc}") from exc
        if response.status_code != 200:
            body = response.text.strip()
            raise RuntimeError(
                f"Ollama generate failed with status {response.status_code}: {body}"
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON response: {response.text.strip()}"
            ) from exc
        output = payload.get("response")
        if not isinstance(output, str):
            raise RuntimeError(
                f"Ollama response missing 'response' field: {json.dumps(payload, ensure_ascii=False)}"
            )
        return output.strip()
