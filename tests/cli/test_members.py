"""Tests for ``swarm members`` command."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx
import yaml
from typer.testing import CliRunner

from src.cli.main import app
from src.cli.utils.config import ConfigManager
from src.state.models import SwarmMember

runner = CliRunner()

SWARM_UUID = "716a4150-ab9d-4b54-a2a8-f2b7c607c21e"
UNKNOWN_SWARM_UUID = "00000000-0000-0000-0000-000000000000"


def _make_member(
    agent_id: str = "alice",
    endpoint: str = "https://alice.example.com/swarm",
    public_key: str = "abcdefgh-very-long-public-key-string",
) -> SwarmMember:
    return SwarmMember(
        agent_id=agent_id,
        endpoint=endpoint,
        public_key=public_key,
        joined_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc),
    )


def _setup_config(tmpdir: str, monkeypatch) -> Path:
    """Create a config dir with default_swarm pointing at SWARM_UUID."""
    config_dir = Path(tmpdir) / "swarm"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "agent_id": "test-agent",
                "endpoint": "https://test.example.com/swarm",
                "default_swarm": SWARM_UUID,
            }
        )
    )
    (config_dir / "agent.key").write_text("dummy-key")
    monkeypatch.setattr(ConfigManager, "DEFAULT_DIR", config_dir)
    return config_dir


class TestMembersListing:
    """Default tabular listing case."""

    def test_basic_listing(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            members = [
                _make_member("alice", "https://alice.example.com/swarm", "AAAAbbbbCCCCddddEEEE"),
                _make_member("bob", "https://bob.example.com/swarm", "FFFFggggHHHHiiiiJJJJ"),
            ]
            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=members),
            ):
                result = runner.invoke(app, ["members"])

            assert result.exit_code == 0
            assert "AGENT_ID" in result.stdout
            assert "ENDPOINT" in result.stdout
            assert "PUBLIC_KEY" in result.stdout
            assert "alice" in result.stdout
            assert "bob" in result.stdout
            assert "https://alice.example.com/swarm" in result.stdout
            # Public key truncated to 8 chars + ellipsis -- full key absent.
            assert "AAAAbbbb" in result.stdout
            assert "AAAAbbbbCCCCddddEEEE" not in result.stdout


class TestMembersJsonOutput:
    """``--json`` returns full data with full public keys."""

    def test_json_output(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            full_pk = "AAAAbbbbCCCCddddEEEEffffGGGGhhhh"
            members = [_make_member("alice", "https://alice.example.com/swarm", full_pk)]
            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=members),
            ):
                result = runner.invoke(app, ["members", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["swarm_id"] == SWARM_UUID
            assert data["member_count"] == 1
            assert len(data["members"]) == 1
            entry = data["members"][0]
            assert entry["agent_id"] == "alice"
            assert entry["endpoint"] == "https://alice.example.com/swarm"
            # JSON path returns the FULL public key, not truncated.
            assert entry["public_key"] == full_pk
            assert "joined_at" in entry
            # No --health, so no health field.
            assert "health" not in entry


class TestMembersEmptySwarm:
    """The swarm exists but has zero members."""

    def test_empty_swarm_table(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=[]),
            ):
                result = runner.invoke(app, ["members"])

            assert result.exit_code == 0
            assert "No members" in result.stdout

    def test_empty_swarm_json(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=[]),
            ):
                result = runner.invoke(app, ["members", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["swarm_id"] == SWARM_UUID
            assert data["member_count"] == 0
            assert data["members"] == []


class TestMembersUnknownSwarmId:
    """``--swarm-id <uuid>`` for a swarm the agent is not a member of."""

    def test_unknown_swarm_exits_nonzero(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            # Repository returns None when the agent isn't a member.
            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=None),
            ):
                result = runner.invoke(
                    app, ["members", "--swarm-id", UNKNOWN_SWARM_UUID]
                )

            assert result.exit_code == 5
            assert "Not a member" in result.stdout
            assert UNKNOWN_SWARM_UUID in result.stdout

    def test_invalid_uuid_exits_with_validation_code(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            result = runner.invoke(app, ["members", "--swarm-id", "not-a-uuid"])

            assert result.exit_code == 2


class TestMembersHealthFlag:
    """``--health`` probes endpoints in parallel; uses mocked HTTP client."""

    def test_health_ok(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            members = [
                _make_member("alice", "https://alice.example.com/swarm"),
                _make_member("bob", "https://bob.example.com/swarm"),
            ]

            class _MockResponse:
                def __init__(self, status: int) -> None:
                    self.status_code = status

            class _MockClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                async def __aenter__(self) -> "_MockClient":
                    return self

                async def __aexit__(self, *args) -> None:
                    return None

                async def get(self, url: str) -> _MockResponse:
                    assert url.endswith("/health")
                    return _MockResponse(200)

            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=members),
            ), patch("src.cli.commands.members.httpx.AsyncClient", _MockClient):
                result = runner.invoke(app, ["members", "--health", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert all(m["health"] == "OK" for m in data["members"])

    def test_health_mixed_statuses(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            members = [
                _make_member("alice", "https://alice.example.com/swarm"),
                _make_member("bob", "https://bob.example.com/swarm"),
                _make_member("carol", "https://carol.example.com/swarm"),
                _make_member("dave", "https://dave.example.com/swarm"),
            ]

            class _Resp:
                def __init__(self, status: int) -> None:
                    self.status_code = status

            class _MockClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                async def __aenter__(self) -> "_MockClient":
                    return self

                async def __aexit__(self, *args) -> None:
                    return None

                async def get(self, url: str) -> _Resp:
                    if "alice" in url:
                        return _Resp(200)
                    if "bob" in url:
                        return _Resp(503)
                    if "carol" in url:
                        raise httpx.TimeoutException("timed out")
                    if "dave" in url:
                        raise httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")
                    raise AssertionError(f"Unexpected url {url}")

            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=members),
            ), patch("src.cli.commands.members.httpx.AsyncClient", _MockClient):
                result = runner.invoke(app, ["members", "--health", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.stdout)
            health = {m["agent_id"]: m["health"] for m in data["members"]}
            assert health["alice"] == "OK"
            assert health["bob"] == "HTTP 503"
            assert health["carol"] == "timeout"
            assert health["dave"] == "TLS error"

    def test_health_table_includes_health_column(self, monkeypatch):
        with TemporaryDirectory() as tmpdir:
            _setup_config(tmpdir, monkeypatch)

            members = [_make_member("alice", "https://alice.example.com/swarm")]

            class _Resp:
                status_code = 200

            class _MockClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                async def __aenter__(self) -> "_MockClient":
                    return self

                async def __aexit__(self, *args) -> None:
                    return None

                async def get(self, url: str) -> _Resp:
                    return _Resp()

            with patch(
                "src.cli.commands.members._load_members",
                new=AsyncMock(return_value=members),
            ), patch("src.cli.commands.members.httpx.AsyncClient", _MockClient):
                result = runner.invoke(app, ["members", "--health"])

            assert result.exit_code == 0
            assert "HEALTH" in result.stdout
            assert "OK" in result.stdout
