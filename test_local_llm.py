import importlib.util
import os
from pathlib import Path

import llm_router

os.environ.setdefault("LOCAL_LLM_ONLY", "1")
os.environ.setdefault("BLACKCLAW_MODEL", "qwen3:8b")
os.environ.setdefault("TAVILY_API_KEY", "test")


def test_llm_router_uses_configurable_timeout_and_num_predict(monkeypatch) -> None:
    captured = {}

    class _Response:
        status_code = 200
        text = ""

        def json(self):
            return {"response": "{\"ok\": true}"}

    def fake_post(url, json=None, timeout=None, **_kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_S", "345")
    monkeypatch.setattr(llm_router.requests, "post", fake_post)

    router = llm_router.LLMRouter(base_url="http://localhost:11434")
    text = router.call_local_chat(
        model="qwen3:8b",
        system_prompt="Return JSON.",
        user_prompt='{"ok": true}',
        temperature=0.2,
        num_predict=512,
    )

    assert text == '{"ok": true}'
    assert captured["timeout"] == 345.0
    assert captured["json"]["options"]["temperature"] == 0.2
    assert captured["json"]["options"]["num_predict"] == 512


def test_ollama_client_forwards_generation_config(monkeypatch) -> None:
    captured = {}

    class _FakeRouter:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def call_local_chat(
            self,
            *,
            model,
            system_prompt,
            user_prompt,
            temperature,
            request_timeout_s=None,
            num_predict=None,
        ):
            captured["model"] = model
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["temperature"] = temperature
            captured["request_timeout_s"] = request_timeout_s
            captured["num_predict"] = num_predict
            return '{"ok": true}'

    module_path = Path(__file__).with_name("llm_client.py")
    spec = importlib.util.spec_from_file_location("real_llm_client_test", module_path)
    assert spec is not None and spec.loader is not None
    real_llm_client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_llm_client)

    monkeypatch.setattr(real_llm_client, "LLMRouter", _FakeRouter)

    client = real_llm_client.OllamaClient("qwen3:8b", "http://localhost:11434")
    response = client.generate_content(
        "prompt body",
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 321,
            "request_timeout_s": 456,
        },
    )

    assert response.text == '{"ok": true}'
    assert captured["base_url"] == "http://localhost:11434"
    assert captured["model"] == "qwen3:8b"
    assert captured["temperature"] == 0.1
    assert captured["request_timeout_s"] == 456
    assert captured["num_predict"] == 321
