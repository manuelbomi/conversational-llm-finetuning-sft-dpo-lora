from fastapi.testclient import TestClient

from src.serve.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_no_adapters_loaded_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULT_ADAPTER_DIR", str(tmp_path / "does-not-exist"))
    with TestClient(app) as client:
        response = client.get("/adapters")
        assert response.json() == {"loaded": [], "active": None}


def test_default_adapter_loads_and_activates(monkeypatch, tiny_adapter_dir):
    monkeypatch.setenv("DEFAULT_ADAPTER_DIR", str(tiny_adapter_dir))
    with TestClient(app) as client:
        response = client.get("/adapters")
        body = response.json()
        assert body["active"] == "default"
        assert "default" in body["loaded"]


def test_classify_returns_a_completion(monkeypatch, tiny_adapter_dir):
    monkeypatch.setenv("DEFAULT_ADAPTER_DIR", str(tiny_adapter_dir))
    with TestClient(app) as client:
        response = client.post(
            "/classify", json={"utterance": "this is too expensive", "max_new_tokens": 8}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["adapter"] == "default"
        assert isinstance(body["raw_completion"], str)
        # A fresh, untrained adapter has no reason to produce a parseable
        # "Label: X" line -- this documents that honestly rather than
        # asserting a specific label.
        assert "label" in body


def test_suggest_response_returns_a_completion(monkeypatch, tiny_adapter_dir):
    monkeypatch.setenv("DEFAULT_ADAPTER_DIR", str(tiny_adapter_dir))
    with TestClient(app) as client:
        response = client.post(
            "/suggest_response", json={"utterance": "how do I get started?", "max_new_tokens": 8}
        )
        assert response.status_code == 200
        assert response.json()["adapter"] == "default"


def test_activate_unknown_adapter_returns_404(monkeypatch, tiny_adapter_dir):
    monkeypatch.setenv("DEFAULT_ADAPTER_DIR", str(tiny_adapter_dir))
    with TestClient(app) as client:
        response = client.post("/adapters/does-not-exist/activate")
        assert response.status_code == 404
