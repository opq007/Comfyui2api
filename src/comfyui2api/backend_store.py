from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ConflictError, DuplicateError, NotFoundError, UrlError, ValidationError
from .slugs import (
    normalize_base_url,
    parse_display_name,
    parse_health_interval_s,
    parse_max_in_flight,
    parse_routing_policy,
    parse_slug,
)
from .util import utc_now_unix


ACTIVE_STATUSES = ("pending", "queued", "running")
RUNNING_STATUS = "running"


@dataclass(frozen=True)
class InstanceRecord:
    slug: str
    display_name: str | None
    base_url: str
    auth_token: str | None
    enabled: bool
    max_in_flight: int
    health_interval_s: int | None
    created_at: int
    updated_at: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "max_in_flight": self.max_in_flight,
            "health_interval_s": self.health_interval_s,
            "has_auth_token": bool(self.auth_token),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ExternalModelRecord:
    slug: str
    display_name: str | None
    workflow_name: str | None
    routing_policy: str
    enabled: bool
    created_at: int
    updated_at: int
    instance_slugs: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "workflow_name": self.workflow_name,
            "routing_policy": self.routing_policy,
            "enabled": self.enabled,
            "instance_slugs": list(self.instance_slugs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BackendStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        await self._run_sync(self._init_sync)

    async def list_instances(self) -> list[InstanceRecord]:
        return await self._run_sync(self._list_instances_sync)

    async def get_instance(self, slug: str) -> InstanceRecord | None:
        return await self._run_sync(self._get_instance_sync, slug)

    async def create_instance(self, payload: dict[str, Any]) -> InstanceRecord:
        return await self._run_sync(self._create_instance_sync, payload)

    async def patch_instance(self, slug: str, payload: dict[str, Any]) -> InstanceRecord:
        return await self._run_sync(self._patch_instance_sync, slug, payload)

    async def delete_instance(self, slug: str) -> None:
        await self._run_sync(self._delete_instance_sync, slug)

    async def list_models(self, *, enabled_only: bool = False) -> list[ExternalModelRecord]:
        return await self._run_sync(self._list_models_sync, enabled_only)

    async def get_model(self, slug: str) -> ExternalModelRecord | None:
        return await self._run_sync(self._get_model_sync, slug)

    async def create_model(self, payload: dict[str, Any]) -> ExternalModelRecord:
        return await self._run_sync(self._create_model_sync, payload)

    async def patch_model(self, slug: str, payload: dict[str, Any]) -> ExternalModelRecord:
        return await self._run_sync(self._patch_model_sync, slug, payload)

    async def delete_model(self, slug: str) -> None:
        await self._run_sync(self._delete_model_sync, slug)

    async def bound_model_count(self, instance_slug: str) -> int:
        return await self._run_sync(self._bound_model_count_sync, instance_slug)

    async def has_running_for_instance(self, instance_slug: str) -> bool:
        return await self._run_sync(self._has_task_sync, "instance_slug", instance_slug, (RUNNING_STATUS,))

    async def has_active_for_model(self, model_slug: str) -> bool:
        return await self._run_sync(self._has_task_sync, "model_slug", model_slug, ACTIVE_STATUSES)

    async def _run_sync(self, fn: Any, *args: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.database_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init_sync(self) -> None:
        with closing(self._connect()) as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS instances (
                    slug TEXT PRIMARY KEY,
                    display_name TEXT,
                    base_url TEXT NOT NULL UNIQUE,
                    auth_token TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    max_in_flight INTEGER NOT NULL DEFAULT 1 CHECK (max_in_flight BETWEEN 1 AND 8),
                    health_interval_s INTEGER,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_models (
                    slug TEXT PRIMARY KEY,
                    display_name TEXT,
                    workflow_name TEXT,
                    routing_policy TEXT NOT NULL DEFAULT 'round_robin',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_instances (
                    model_slug TEXT NOT NULL,
                    instance_slug TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (model_slug, instance_slug),
                    FOREIGN KEY(model_slug) REFERENCES external_models(slug) ON DELETE CASCADE,
                    FOREIGN KEY(instance_slug) REFERENCES instances(slug) ON DELETE CASCADE
                );
                """
            )
            con.commit()

    def _list_instances_sync(self) -> list[InstanceRecord]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM instances ORDER BY slug COLLATE BINARY").fetchall()
        return [_instance_from_row(row) for row in rows]

    def _get_instance_sync(self, slug: str) -> InstanceRecord | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM instances WHERE slug = ?", (slug,)).fetchone()
        return _instance_from_row(row) if row else None

    def _create_instance_sync(self, payload: dict[str, Any]) -> InstanceRecord:
        slug = parse_slug(str(payload.get("slug") or ""))
        display_name = parse_display_name(payload.get("display_name"))
        try:
            base_url = normalize_base_url(str(payload.get("base_url") or ""))
        except UrlError as exc:
            raise ValidationError(str(exc)) from exc
        auth_token = _optional_token(payload.get("auth_token"), missing_ok=True)
        enabled = _as_bool(payload.get("enabled"), default=True)
        max_in_flight = parse_max_in_flight(payload.get("max_in_flight"))
        health_interval_s = parse_health_interval_s(payload.get("health_interval_s"))
        now = utc_now_unix()
        with closing(self._connect()) as con:
            if con.execute("SELECT 1 FROM instances WHERE slug = ?", (slug,)).fetchone():
                raise DuplicateError(f"Instance slug `{slug}` already exists.")
            if con.execute("SELECT 1 FROM instances WHERE base_url = ?", (base_url,)).fetchone():
                raise DuplicateError(f"Instance base_url `{base_url}` already exists.")
            con.execute(
                """
                INSERT INTO instances (
                    slug, display_name, base_url, auth_token, enabled, max_in_flight,
                    health_interval_s, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    display_name,
                    base_url,
                    auth_token,
                    1 if enabled else 0,
                    max_in_flight,
                    health_interval_s,
                    now,
                    now,
                ),
            )
            con.commit()
        record = self._get_instance_sync(slug)
        assert record is not None
        return record

    def _patch_instance_sync(self, slug: str, payload: dict[str, Any]) -> InstanceRecord:
        current = self._get_instance_sync(slug)
        if current is None:
            raise NotFoundError(f"Instance `{slug}` not found.")
        display_name = current.display_name
        if "display_name" in payload:
            display_name = parse_display_name(payload.get("display_name"))
        base_url = current.base_url
        if "base_url" in payload:
            try:
                base_url = normalize_base_url(str(payload.get("base_url") or ""))
            except UrlError as exc:
                raise ValidationError(str(exc)) from exc
            if base_url != current.base_url and self._has_task_sync("instance_slug", slug, (RUNNING_STATUS,)):
                raise ConflictError("Cannot change base_url while the instance has a running job.")
        auth_token = current.auth_token
        if "auth_token" in payload:
            auth_token = _optional_token(payload.get("auth_token"), missing_ok=False)
        enabled = current.enabled
        if "enabled" in payload:
            enabled = _as_bool(payload.get("enabled"), default=current.enabled)
        max_in_flight = current.max_in_flight
        if "max_in_flight" in payload:
            max_in_flight = parse_max_in_flight(payload.get("max_in_flight"))
        health_interval_s = current.health_interval_s
        if "health_interval_s" in payload:
            health_interval_s = parse_health_interval_s(payload.get("health_interval_s"))
        now = utc_now_unix()
        with closing(self._connect()) as con:
            if base_url != current.base_url:
                other = con.execute(
                    "SELECT slug FROM instances WHERE base_url = ? AND slug != ?",
                    (base_url, slug),
                ).fetchone()
                if other:
                    raise DuplicateError(f"Instance base_url `{base_url}` already exists.")
            con.execute(
                """
                UPDATE instances
                SET display_name = ?, base_url = ?, auth_token = ?, enabled = ?, max_in_flight = ?,
                    health_interval_s = ?, updated_at = ?
                WHERE slug = ?
                """,
                (
                    display_name,
                    base_url,
                    auth_token,
                    1 if enabled else 0,
                    max_in_flight,
                    health_interval_s,
                    now,
                    slug,
                ),
            )
            con.commit()
        record = self._get_instance_sync(slug)
        assert record is not None
        return record

    def _delete_instance_sync(self, slug: str) -> None:
        current = self._get_instance_sync(slug)
        if current is None:
            raise NotFoundError(f"Instance `{slug}` not found.")
        if self._has_task_sync("instance_slug", slug, (RUNNING_STATUS,)):
            raise ConflictError("Cannot delete instance while a job is running on it.")
        with closing(self._connect()) as con:
            con.execute("DELETE FROM model_instances WHERE instance_slug = ?", (slug,))
            con.execute("DELETE FROM instances WHERE slug = ?", (slug,))
            con.commit()

    def _list_models_sync(self, enabled_only: bool) -> list[ExternalModelRecord]:
        sql = "SELECT * FROM external_models"
        params: list[Any] = []
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY slug COLLATE BINARY"
        with closing(self._connect()) as con:
            rows = con.execute(sql, params).fetchall()
            bindings = _bindings_by_model(con)
        return [_model_from_row(row, bindings.get(str(row["slug"]), ())) for row in rows]

    def _get_model_sync(self, slug: str) -> ExternalModelRecord | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM external_models WHERE slug = ?", (slug,)).fetchone()
            if row is None:
                return None
            bindings = _bindings_by_model(con).get(slug, ())
        return _model_from_row(row, bindings)

    def _create_model_sync(self, payload: dict[str, Any]) -> ExternalModelRecord:
        slug = parse_slug(str(payload.get("slug") or ""))
        display_name = parse_display_name(payload.get("display_name"))
        workflow_name = _optional_text(payload.get("workflow_name"))
        routing_policy = parse_routing_policy(payload.get("routing_policy"))
        enabled = _as_bool(payload.get("enabled"), default=False)
        instance_slugs = _parse_instance_slugs(payload.get("instance_slugs"))
        now = utc_now_unix()
        with closing(self._connect()) as con:
            if con.execute("SELECT 1 FROM external_models WHERE slug = ?", (slug,)).fetchone():
                raise DuplicateError(f"Model slug `{slug}` already exists.")
            _assert_instances_exist(con, instance_slugs)
            con.execute(
                """
                INSERT INTO external_models (
                    slug, display_name, workflow_name, routing_policy, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (slug, display_name, workflow_name, routing_policy, 1 if enabled else 0, now, now),
            )
            _replace_bindings(con, slug, instance_slugs, now)
            con.commit()
        record = self._get_model_sync(slug)
        assert record is not None
        return record

    def _patch_model_sync(self, slug: str, payload: dict[str, Any]) -> ExternalModelRecord:
        current = self._get_model_sync(slug)
        if current is None:
            raise NotFoundError(f"Model `{slug}` not found.")
        display_name = current.display_name
        if "display_name" in payload:
            display_name = parse_display_name(payload.get("display_name"))
        workflow_name = current.workflow_name
        if "workflow_name" in payload:
            workflow_name = _optional_text(payload.get("workflow_name"))
        routing_policy = current.routing_policy
        if "routing_policy" in payload:
            routing_policy = parse_routing_policy(payload.get("routing_policy"))
        enabled = current.enabled
        if "enabled" in payload:
            enabled = _as_bool(payload.get("enabled"), default=current.enabled)
        instance_slugs = list(current.instance_slugs)
        if "instance_slugs" in payload:
            instance_slugs = _parse_instance_slugs(payload.get("instance_slugs"))
        now = utc_now_unix()
        with closing(self._connect()) as con:
            _assert_instances_exist(con, instance_slugs)
            con.execute(
                """
                UPDATE external_models
                SET display_name = ?, workflow_name = ?, routing_policy = ?, enabled = ?, updated_at = ?
                WHERE slug = ?
                """,
                (display_name, workflow_name, routing_policy, 1 if enabled else 0, now, slug),
            )
            if "instance_slugs" in payload:
                _replace_bindings(con, slug, instance_slugs, now)
            con.commit()
        record = self._get_model_sync(slug)
        assert record is not None
        return record

    def _delete_model_sync(self, slug: str) -> None:
        current = self._get_model_sync(slug)
        if current is None:
            raise NotFoundError(f"Model `{slug}` not found.")
        if self._has_task_sync("model_slug", slug, ACTIVE_STATUSES):
            raise ConflictError("Cannot delete model while jobs are pending, queued, or running.")
        with closing(self._connect()) as con:
            con.execute("DELETE FROM model_instances WHERE model_slug = ?", (slug,))
            con.execute("DELETE FROM external_models WHERE slug = ?", (slug,))
            con.commit()

    def _bound_model_count_sync(self, instance_slug: str) -> int:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM model_instances WHERE instance_slug = ?",
                (instance_slug,),
            ).fetchone()
        return int(row[0] if row else 0)

    def _has_task_sync(self, column: str, slug: str, statuses: Iterable[str]) -> bool:
        status_list = list(statuses)
        placeholders = ",".join("?" for _ in status_list)
        # A missing tasks table/column legitimately means "no running jobs".
        # PRAGMA never raises for a missing table (it just returns no rows).
        # Anything else (e.g. "database is locked") propagates: swallowing it
        # here used to turn a failed lookup into "no running jobs", which let
        # instance deletion / base_url changes bypass the running-job guard.
        with closing(self._connect()) as con:
            cols = {str(item[1]) for item in con.execute("PRAGMA table_info(tasks)").fetchall()}
            if column not in cols:
                return False
            row = con.execute(
                f"SELECT 1 FROM tasks WHERE {column} = ? AND status IN ({placeholders}) LIMIT 1",
                (slug, *status_list),
            ).fetchone()
        return row is not None


def _instance_from_row(row: sqlite3.Row) -> InstanceRecord:
    return InstanceRecord(
        slug=str(row["slug"]),
        display_name=row["display_name"],
        base_url=str(row["base_url"]),
        auth_token=row["auth_token"],
        enabled=bool(row["enabled"]),
        max_in_flight=int(row["max_in_flight"]),
        health_interval_s=row["health_interval_s"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


def _model_from_row(row: sqlite3.Row, instance_slugs: tuple[str, ...]) -> ExternalModelRecord:
    return ExternalModelRecord(
        slug=str(row["slug"]),
        display_name=row["display_name"],
        workflow_name=row["workflow_name"],
        routing_policy=str(row["routing_policy"]),
        enabled=bool(row["enabled"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        instance_slugs=tuple(sorted(instance_slugs)),
    )


def _bindings_by_model(con: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    rows = con.execute(
        "SELECT model_slug, instance_slug FROM model_instances ORDER BY instance_slug COLLATE BINARY"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(str(row["model_slug"]), []).append(str(row["instance_slug"]))
    return {key: tuple(values) for key, values in out.items()}


def _replace_bindings(con: sqlite3.Connection, model_slug: str, instance_slugs: list[str], now: int) -> None:
    con.execute("DELETE FROM model_instances WHERE model_slug = ?", (model_slug,))
    con.executemany(
        "INSERT INTO model_instances (model_slug, instance_slug, created_at) VALUES (?, ?, ?)",
        [(model_slug, slug, now) for slug in instance_slugs],
    )


def _assert_instances_exist(con: sqlite3.Connection, slugs: list[str]) -> None:
    for slug in slugs:
        if con.execute("SELECT 1 FROM instances WHERE slug = ?", (slug,)).fetchone() is None:
            raise ValidationError(f"Unknown instance slug `{slug}`.")


def _parse_instance_slugs(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError("instance_slugs must be an array of slugs")
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        slug = parse_slug(str(item or ""))
        if slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out


def _optional_text(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _optional_token(raw: Any, *, missing_ok: bool) -> str | None:
    if raw is None:
        return None if not missing_ok else None
    text = str(raw).strip()
    return text or None


def _as_bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValidationError("enabled must be a boolean")
