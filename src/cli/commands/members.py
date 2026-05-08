"""List members of a swarm and their endpoints.

Reads ``swarm_members`` via the state-layer ``MembershipRepository`` (no raw
SQL in the CLI).  Optionally probes each member's ``/health`` endpoint in
parallel and reports per-member status.
"""

import asyncio
from uuid import UUID

import httpx
import typer
from rich.console import Console

from src.cli.output import format_error, format_table, json_output
from src.cli.utils import ConfigManager, resolve_swarm_id, SwarmIdError
from src.cli.utils.config import ConfigError
from src.state import DatabaseManager, MembershipRepository
from src.state.models import SwarmMember

console = Console()

_HEALTH_TIMEOUT = 5.0
_PUBKEY_TRUNC_CHARS = 8


async def _load_members(swarm_id: UUID) -> list[SwarmMember]:
    """Load members for ``swarm_id`` via the membership repository.

    Returns an empty list when the agent is a member of the swarm but the
    swarm has no other members; returns ``None`` when the agent is not a
    member of the requested swarm at all (used by the caller to distinguish
    "empty swarm" from "not a member").
    """
    config = ConfigManager()
    agent_config = config.load()

    db = DatabaseManager(agent_config.db_path)
    await db.initialize()

    async with db.connection() as conn:
        repo = MembershipRepository(conn)
        membership = await repo.get_swarm(str(swarm_id))

    if membership is None:
        return None
    return list(membership.members)


async def _probe_member(member: SwarmMember) -> str:
    """TLS handshake + GET ``<endpoint>/health``; return short status string.

    Uses the agent's configured ``endpoint`` directly (it already includes
    the path component, e.g. ``https://host/swarm``); the health probe hits
    ``<endpoint>/health`` regardless of the path's exact shape, matching the
    contract documented in the originating issue.
    """
    url = member.endpoint.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            return "OK"
        return f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except (
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
    ) as e:
        # SSL/TLS errors are wrapped under ConnectError on httpx; surface
        # the underlying class name where it helps disambiguate.
        cause = e.__cause__
        if cause is not None and "SSL" in type(cause).__name__.upper():
            return "TLS error"
        if "SSL" in str(e).upper() or "CERTIFICATE" in str(e).upper():
            return "TLS error"
        return f"connect error: {type(e).__name__}"
    except Exception as e:  # pragma: no cover - defensive
        return f"error: {type(e).__name__}"


async def _probe_all(members: list[SwarmMember]) -> list[str]:
    """Probe every member in parallel; preserve input order."""
    return await asyncio.gather(*(_probe_member(m) for m in members))


def _truncate_pubkey(pk: str, n: int = _PUBKEY_TRUNC_CHARS) -> str:
    """Return ``pk`` truncated to ``n`` chars + ellipsis when longer."""
    if len(pk) <= n:
        return pk
    return f"{pk[:n]}…"


def members_command(
    swarm_id: str | None,
    health: bool,
    json_flag: bool,
) -> None:
    """List members of a swarm and their endpoints."""
    try:
        swarm_uuid = resolve_swarm_id(swarm_id)
    except SwarmIdError as e:
        format_error(console, str(e))
        raise typer.Exit(code=2)
    except ValueError as e:
        format_error(console, str(e))
        raise typer.Exit(code=2)

    try:
        members = asyncio.run(_load_members(swarm_uuid))
    except ConfigError as e:
        format_error(console, str(e), hint="Run 'swarm init' to configure your agent")
        raise typer.Exit(code=1)
    except Exception as e:
        format_error(console, f"Failed to load members: {e}")
        raise typer.Exit(code=1)

    if members is None:
        format_error(
            console,
            f"Not a member of swarm {swarm_uuid}",
            hint="Use 'swarm list' to see swarms this agent belongs to",
        )
        raise typer.Exit(code=5)

    statuses: list[str] = []
    if health and members:
        try:
            statuses = asyncio.run(_probe_all(members))
        except Exception as e:
            format_error(console, f"Health probe failed: {e}")
            raise typer.Exit(code=1)

    if json_flag:
        data = {
            "swarm_id": str(swarm_uuid),
            "member_count": len(members),
            "members": [
                {
                    "agent_id": m.agent_id,
                    "endpoint": m.endpoint,
                    "public_key": m.public_key,
                    "joined_at": m.joined_at.isoformat(),
                    **({"health": statuses[i]} if health else {}),
                }
                for i, m in enumerate(members)
            ],
        }
        json_output(console, data)
        return

    if not members:
        console.print(f"[dim]No members in swarm {swarm_uuid}.[/dim]")
        return

    columns = ["AGENT_ID", "ENDPOINT", "PUBLIC_KEY"]
    if health:
        columns.append("HEALTH")
    rows: list[list[str]] = []
    for i, m in enumerate(members):
        row = [m.agent_id, m.endpoint, _truncate_pubkey(m.public_key)]
        if health:
            row.append(statuses[i])
        rows.append(row)
    format_table(console, f"Members of swarm {swarm_uuid}", columns, rows)
