"""LLM istemci timeout'u: httpbin'in REQUEST_TIMEOUT'undan ayri ve gercekten etkili.

Test GERCEK bir yerel HTTP sunucusuna karsi kosar; sunucu yaniti bilerek
geciktirir. Boylece timeout'un mock'lanmis degil, ag katmaninda uygulandigi
kanitlanir.

OLCEK NOTU: sartnamedeki 15 sn gecikme / 30 sn gecer / 5 sn patlar senaryosu
oranlari korunarak kuculdu (SLOW_RESPONSE_S / GENEROUS_S / TIGHT_S), cunku
20 saniyelik bir test tum paketi yavaslatir. Kanitlanan davranis aynidir.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import config
import llm_timeout
from error_taxonomy import classify_error, failure_origin
from generators.groq_gen import GroqGenerator

SLOW_RESPONSE_S = 1.5   # sartnamedeki 15 sn'nin olcekli karsiligi
GENEROUS_S = 5.0        # 30 sn karsiligi  -> gecmeli
TIGHT_S = 0.4           # 5 sn karsiligi   -> timeout uretmeli

_BODY = {
    "id": "chatcmpl-test",
    "model": "gpt-test",
    "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler arayuzu
        time.sleep(SLOW_RESPONSE_S)
        payload = json.dumps(_BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # sessiz
        return


@pytest.fixture
def slow_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


class _LocalGenerator(GroqGenerator):
    """Groq uyarlayicisi; yalnizca base_url yerel sunucuya cevrilir."""


def _client_for(base_url, monkeypatch, read_timeout):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        config, "LLM_REQUEST_TIMEOUT_BY_PROVIDER",
        {"groq": {"connect": 2.0, "read": read_timeout, "write": read_timeout, "pool": read_timeout}},
    )
    gen = _LocalGenerator("gpt-test")
    gen._base_url = base_url
    return gen


def test_generous_timeout_lets_slow_response_through(slow_server, monkeypatch):
    gen = _client_for(slow_server, monkeypatch, GENEROUS_S)
    text, usage = gen._request_completion("prompt", 64)
    assert text == "[]"
    assert usage.split_available is True


def test_tight_timeout_produces_infrastructure_timeout(slow_server, monkeypatch):
    gen = _client_for(slow_server, monkeypatch, TIGHT_S)
    with pytest.raises(Exception) as excinfo:
        gen._request_completion("prompt", 64)

    error_class = classify_error(excinfo.value)
    assert error_class == "TIMEOUT", f"beklenen TIMEOUT, gelen {error_class}"
    assert failure_origin(error_class) == "altyapi"
    assert llm_timeout.timeout_phase(excinfo.value) == "read"


def test_ledger_records_timeout_seconds_and_phase(slow_server, monkeypatch, tmp_path):
    from call_ledger import CallLedger

    gen = _client_for(slow_server, monkeypatch, TIGHT_S)
    with pytest.raises(Exception) as excinfo:
        gen._request_completion("prompt", 64)

    ledger = CallLedger(str(tmp_path), run_id="r")
    ledger.record_failed_call(
        generator="LLM-Groq-gpt-test", variant="basic", operation_id="EP1",
        method="GET", path="/a", attempt=0, call_type="ana", exc=excinfo.value,
        provider_label="Groq",
    )
    ledger.flush()
    record = CallLedger.load(ledger.path)[0]
    assert record["error_class"] == "TIMEOUT"
    assert record["failure_origin"] == "altyapi"
    assert record["timeout_phase"] == "read"
    assert record["timeout_seconds"]["read"] == TIGHT_S


# ── httpbin timeout'undan ayrilma ────────────────────────────────────────

def test_llm_timeout_is_separate_from_httpbin_timeout():
    """runner.py'nin 10 sn'lik REQUEST_TIMEOUT'u LLM'e uygulanmamali."""
    assert config.REQUEST_TIMEOUT == 10
    for provider in ("OpenAI", "Groq", "Gemini", "Claude"):
        assert llm_timeout.timeout_seconds_for(provider)["read"] > config.REQUEST_TIMEOUT


def test_default_is_not_below_any_sdk_default():
    """Varsayilanimiz kurulu SDK'larin varsayilanlarinin altina inmemeli.

    openai 1.109.1 ve anthropic 0.120.2: connect=5, read=600.
    google-genai 1.47.0: timeout yok (sonsuz) -> alt sinir koymaz.
    """
    import anthropic._constants as ac
    import openai._constants as oc

    sdk_connect = min(oc.DEFAULT_TIMEOUT.connect, ac.DEFAULT_TIMEOUT.connect)
    sdk_read = min(oc.DEFAULT_TIMEOUT.read, ac.DEFAULT_TIMEOUT.read)
    ours = config.LLM_REQUEST_TIMEOUT
    assert ours["connect"] >= sdk_connect
    assert ours["read"] >= sdk_read


def test_provider_override_wins():
    original = dict(config.LLM_REQUEST_TIMEOUT_BY_PROVIDER)
    try:
        config.LLM_REQUEST_TIMEOUT_BY_PROVIDER["groq"] = {"connect": 1.0, "read": 2.0}
        assert llm_timeout.timeout_seconds_for("Groq")["read"] == 2.0
        assert llm_timeout.timeout_seconds_for("OpenAI")["read"] == config.LLM_REQUEST_TIMEOUT["read"]
    finally:
        config.LLM_REQUEST_TIMEOUT_BY_PROVIDER.clear()
        config.LLM_REQUEST_TIMEOUT_BY_PROVIDER.update(original)


def test_genai_timeout_is_milliseconds():
    """google-genai HttpOptions.timeout MILISANIYE bekler (_api_client.py:208-210)."""
    assert llm_timeout.genai_timeout_ms_for("Gemini") == int(
        config.LLM_REQUEST_TIMEOUT["read"] * 1000
    )
