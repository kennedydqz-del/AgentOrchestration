"""Tests for the Orchestrator API client SDK."""

import json
from io import BytesIO
from urllib.error import HTTPError

from src.sdk.client import OrchestratorClient


class TestOrchestratorClient:
    def test_default_base_url(self):
        client = OrchestratorClient(api_key="test")
        assert client.base_url == "https://api.agent-orchestrator.io"

    def test_custom_base_url(self):
        client = OrchestratorClient(base_url="https://custom.example.com", api_key="test")
        assert client.base_url == "https://custom.example.com"

    def test_base_url_strips_trailing_slash(self):
        client = OrchestratorClient(
            base_url="https://api.agent-orchestrator.io/", api_key="test"
        )
        assert client.base_url == "https://api.agent-orchestrator.io"

    def test_base_url_strips_multiple_trailing_slashes(self):
        client = OrchestratorClient(
            base_url="https://api.agent-orchestrator.io///", api_key="test"
        )
        assert client.base_url == "https://api.agent-orchestrator.io"

    def test_request_builds_correct_url_without_trailing_slash(self, monkeypatch):
        client = OrchestratorClient(
            base_url="https://api.agent-orchestrator.io/", api_key="test-key"
        )
        captured = {}

        def mock_urlopen(req, **kw):
            captured["url"] = req.full_url
            return BytesIO(json.dumps({"ok": True}).encode())

        monkeypatch.setattr("src.sdk.client.urlopen", mock_urlopen)
        client.list_agents()
        assert captured["url"] == "https://api.agent-orchestrator.io/api/v2/agents"

    def test_request_builds_correct_url_without_trailing_slash_already_clean(self, monkeypatch):
        client = OrchestratorClient(
            base_url="https://api.agent-orchestrator.io", api_key="test-key"
        )
        captured = {}

        def mock_urlopen(req, **kw):
            captured["url"] = req.full_url
            return BytesIO(json.dumps({"ok": True}).encode())

        monkeypatch.setattr("src.sdk.client.urlopen", mock_urlopen)
        client.list_agents()
        assert captured["url"] == "https://api.agent-orchestrator.io/api/v2/agents"
