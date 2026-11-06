from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .jobs import Job, JobOutput
from .util import guess_media_type, utc_now_iso, utc_now_unix


STATUSES = ("pending", "queued", "running", "completed", "failed")

# Playground submissions kept in the ring buffer; older rows are pruned on
# every insert so the history table cannot grow unbounded.
SUBMISSION_RETENTION = 200


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        await self._run_sync(self._init_sync)

    async def upsert_job(self, job: Job) -> None:
        await self._run_sync(self._upsert_job_sync, job)

    async def replace_outputs(self, job_id: str, outputs: list[JobOutput]) -> None:
        await self._run_sync(self._replace_outputs_sync, job_id, outputs)

    async def get_task(self, job_id: str) -> dict[str, Any] | None:
        return await self._run_sync(self._get_task_sync, job_id)

    async def list_tasks(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        q: str | None = None,
        statuses: list[str] | None = None,
        kinds: list[str] | None = None,
        platforms: list[str] | None = None,
        has_outputs: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await self._run_sync(
            self._list_tasks_sync,
            start,
            end,
            q,
            statuses or None,
            kinds or None,
            platforms or None,
            bool(has_outputs),
            max(1, min(200, int(limit))),
            max(0, int(offset)),
        )

    async def stats(self) -> dict[str, Any]:
        return await self._run_sync(self._stats_sync)

    async def mark_unfinished_interrupted(self) -> None:
        await self._run_sync(self._mark_unfinished_interrupted_sync)

    async def add_submission(self, *, kind: str, model: str, params: dict[str, Any], job_id: str) -> dict[str, Any]:
        return await self._run_sync(self._add_submission_sync, kind, model, params, job_id)

    async def list_submissions(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return await self._run_sync(
            self._list_submissions_sync,
            max(1, min(200, int(limit))),
            max(0, int(offset)),
        )

    async def clear_submissions(self) -> dict[str, Any]:
        return await self._run_sync(self._clear_submissions_sync)

    async def _run_sync(self, fn: Any, *args: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    def _connect(self) -> sqlite3.Connection:
        # Reuse a single connection instead of opening + closing one per call.
        # All DB ops are serialized behind `self._lock` (see `_run_sync`), so a
        # persistent connection is not concurrently used. `check_same_thread=False`
        # lets the small to_thread pool share it without a per-call connection.
        if self._conn is None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.database_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    async def aclose(self) -> None:
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _init_sync(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    job_id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    started_at_utc TEXT,
                    finished_at_utc TEXT,
                    updated_at_utc TEXT,
                    duration_s INTEGER,

                    platform TEXT NOT NULL DEFAULT 'Native',
                    kind TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    requested_model TEXT,
                    model_slug TEXT,
                    instance_slug TEXT,

                    status TEXT NOT NULL,
                    progress_percent INTEGER NOT NULL DEFAULT 0,
                    progress_json TEXT,

                    prompt_id TEXT,
                    queue_number INTEGER,
                    current_node TEXT,

                    url TEXT,
                    output_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,

                    prompt_preview TEXT,
                    request_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_kind ON tasks(kind);
                CREATE INDEX IF NOT EXISTS idx_tasks_platform ON tasks(platform);

                CREATE TABLE IF NOT EXISTS task_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    url TEXT NOT NULL,
                    media_type TEXT,
                    node_id TEXT,
                    output_key TEXT,
                    FOREIGN KEY(job_id) REFERENCES tasks(job_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_task_outputs_job_id ON task_outputs(job_id);

                CREATE TABLE IF NOT EXISTS playground_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    model TEXT,
                    params_json TEXT NOT NULL,
                    job_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_pg_submissions_created ON playground_submissions(created_at);
                """
            )
            _ensure_column(con, "tasks", "model_slug", "TEXT")
            _ensure_column(con, "tasks", "instance_slug", "TEXT")
            # Full, never-truncated prompts. `prompt_preview` stays as the short
            # display snippet; rows written before this migration keep NULL here
            # and the UI falls back to the truncated preview for them.
            _ensure_column(con, "tasks", "prompt_full", "TEXT")
            _ensure_column(con, "tasks", "negative_prompt_full", "TEXT")
            con.commit()

    def _upsert_job_sync(self, job: Job) -> None:
        payload = _job_to_row(job)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO tasks (
                    job_id, created_at, created_at_utc, started_at_utc, finished_at_utc,
                    updated_at_utc, duration_s, platform, kind, workflow, requested_model,
                    model_slug, instance_slug, status, progress_percent, progress_json,
                    prompt_id, queue_number, current_node, url, output_count, error,
                    prompt_preview, request_json, prompt_full, negative_prompt_full
                ) VALUES (
                    :job_id, :created_at, :created_at_utc, :started_at_utc, :finished_at_utc,
                    :updated_at_utc, :duration_s, :platform, :kind, :workflow, :requested_model,
                    :model_slug, :instance_slug, :status, :progress_percent, :progress_json,
                    :prompt_id, :queue_number, :current_node, :url, :output_count, :error,
                    :prompt_preview, :request_json, :prompt_full, :negative_prompt_full
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    started_at_utc=excluded.started_at_utc,
                    finished_at_utc=excluded.finished_at_utc,
                    updated_at_utc=excluded.updated_at_utc,
                    duration_s=excluded.duration_s,
                    platform=excluded.platform,
                    kind=excluded.kind,
                    workflow=excluded.workflow,
                    requested_model=excluded.requested_model,
                    model_slug=excluded.model_slug,
                    instance_slug=excluded.instance_slug,
                    status=excluded.status,
                    progress_percent=excluded.progress_percent,
                    progress_json=excluded.progress_json,
                    prompt_id=excluded.prompt_id,
                    queue_number=excluded.queue_number,
                    current_node=excluded.current_node,
                    url=excluded.url,
                    output_count=excluded.output_count,
                    error=excluded.error,
                    prompt_preview=excluded.prompt_preview,
                    request_json=excluded.request_json,
                    prompt_full=excluded.prompt_full,
                    negative_prompt_full=excluded.negative_prompt_full
                """,
                payload,
            )
            con.commit()

    def _replace_outputs_sync(self, job_id: str, outputs: list[JobOutput]) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM task_outputs WHERE job_id = ?", (job_id,))
            con.executemany(
                """
                INSERT INTO task_outputs (job_id, filename, url, media_type, node_id, output_key)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        output.filename,
                        output.url,
                        output.media_type,
                        output.node_id,
                        output.output_key,
                    )
                    for output in outputs
                ],
            )
            con.execute("UPDATE tasks SET output_count = ? WHERE job_id = ?", (len(outputs), job_id))
            con.commit()

    def _get_task_sync(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM tasks WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            outputs = con.execute(
                """
                SELECT filename, url, media_type, node_id, output_key
                FROM task_outputs
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        return {"task": _task_from_row(row), "outputs": [_output_from_row(item) for item in outputs]}

    def _list_tasks_sync(
        self,
        start: str | None,
        end: str | None,
        q: str | None,
        statuses: list[str] | None,
        kinds: list[str] | None,
        platforms: list[str] | None,
        has_outputs: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        where, params = _build_filters(
            start=start,
            end=end,
            q=q,
            statuses=statuses,
            kinds=kinds,
            platforms=platforms,
            has_outputs=has_outputs,
        )
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as con:
            total = int(con.execute(f"SELECT COUNT(*) FROM tasks{where_sql}", params).fetchone()[0])
            rows = con.execute(
                f"""
                SELECT * FROM tasks{where_sql}
                ORDER BY created_at DESC, job_id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            count_rows = con.execute(
                f"SELECT status, COUNT(*) AS count FROM tasks{where_sql} GROUP BY status",
                params,
            ).fetchall()

        counts = {status: 0 for status in STATUSES}
        for row in count_rows:
            counts[str(row["status"])] = int(row["count"])
        items = []
        for row in rows:
            item = _task_from_row(row) | {"outputs": []}
            # Media type of the primary output, derived from its filename so the
            # dashboard can render thumbnails without a per-task detail fetch.
            primary_name = Path(str(item.get("url") or "")).name
            if primary_name:
                item["media_type"] = guess_media_type(primary_name)
            items.append(item)
        return {"total": total, "counts": counts, "items": items}

    def _stats_sync(self) -> dict[str, Any]:
        with self._connect() as con:
            rows = con.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        counts = {status: 0 for status in STATUSES}
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return {"counts": counts}

    def _mark_unfinished_interrupted_sync(self) -> None:
        now_iso = utc_now_iso()
        now_unix = utc_now_unix()
        with self._connect() as con:
            con.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    error = 'Task interrupted by server restart.',
                    finished_at_utc = ?,
                    updated_at_utc = ?,
                    duration_s = MAX(0, ? - created_at),
                    progress_percent = 100
                WHERE status IN ('pending', 'queued', 'running')
                """,
                (now_iso, now_iso, now_unix),
            )
            con.commit()

    def _add_submission_sync(self, kind: str, model: str, params: dict[str, Any], job_id: str) -> dict[str, Any]:
        job_id = str(job_id or "").strip()
        if not job_id:
            raise ValueError("job_id is required for a playground submission")
        now_unix = utc_now_unix()
        now_iso = utc_now_iso()
        with self._connect() as con:
            cursor = con.execute(
                """
                INSERT INTO playground_submissions (created_at, created_at_utc, kind, model, params_json, job_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    now_unix,
                    now_iso,
                    str(kind or ""),
                    str(model or "") or None,
                    json.dumps(params or {}, ensure_ascii=False),
                    job_id,
                ),
            )
            # Ring buffer: drop everything beyond the newest SUBMISSION_RETENTION rows.
            con.execute(
                """
                DELETE FROM playground_submissions
                WHERE id NOT IN (
                    SELECT id FROM playground_submissions ORDER BY id DESC LIMIT ?
                )
                """,
                (SUBMISSION_RETENTION,),
            )
            con.commit()
            return self._submission_by_id(con, int(cursor.lastrowid))

    def _submission_by_id(self, con: sqlite3.Connection, submission_id: int) -> dict[str, Any]:
        row = con.execute(
            """
            SELECT s.id, s.created_at, s.created_at_utc, s.kind, s.model, s.params_json, s.job_id,
                   t.status AS task_status, t.url AS task_url, t.error AS task_error
            FROM playground_submissions s
            LEFT JOIN tasks t ON t.job_id = s.job_id
            WHERE s.id = ?
            """,
            (submission_id,),
        ).fetchone()
        return _submission_from_row(row)

    def _list_submissions_sync(self, limit: int, offset: int) -> dict[str, Any]:
        with self._connect() as con:
            total = int(con.execute("SELECT COUNT(*) FROM playground_submissions").fetchone()[0])
            rows = con.execute(
                """
                SELECT s.id, s.created_at, s.created_at_utc, s.kind, s.model, s.params_json, s.job_id,
                       t.status AS task_status, t.url AS task_url, t.error AS task_error
                FROM playground_submissions s
                LEFT JOIN tasks t ON t.job_id = s.job_id
                ORDER BY s.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        items = [_submission_from_row(row) for row in rows]
        return {"total": total, "items": items}

    def _clear_submissions_sync(self) -> dict[str, Any]:
        with self._connect() as con:
            cursor = con.execute("DELETE FROM playground_submissions")
            con.commit()
            return {"deleted": int(cursor.rowcount)}


def _job_to_row(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "created_at": int(job.created_at),
        "created_at_utc": job.created_at_utc,
        "started_at_utc": job.started_at_utc or None,
        "finished_at_utc": job.finished_at_utc or None,
        "updated_at_utc": job.updated_at_utc or None,
        "duration_s": job.duration_s,
        "platform": job.platform or "Native",
        "kind": job.kind,
        "workflow": job.workflow,
        "requested_model": job.requested_model or job.model_slug or None,
        "model_slug": job.model_slug or job.requested_model or None,
        "instance_slug": job.instance_slug or None,
        "status": job.status,
        "progress_percent": int(job.progress_percent or 0),
        "progress_json": json.dumps(job.progress or {}, ensure_ascii=False),
        "prompt_id": job.prompt_id or None,
        "queue_number": job.queue_number,
        "current_node": job.current_node or None,
        "url": job.url or None,
        "output_count": len(job.outputs or []),
        "error": job.error or None,
        "prompt_preview": _truncate(job.prompt or ""),
        "request_json": json.dumps(job.request_json or {}, ensure_ascii=False),
        "prompt_full": job.prompt or None,
        "negative_prompt_full": job.negative_prompt or None,
    }


def _build_filters(
    *,
    start: str | None,
    end: str | None,
    q: str | None,
    statuses: list[str] | None,
    kinds: list[str] | None,
    platforms: list[str] | None,
    has_outputs: bool = False,
) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if start:
        where.append("created_at_utc >= ?")
        params.append(start)
    if end:
        where.append("created_at_utc <= ?")
        params.append(end)
    if q:
        like = f"%{q}%"
        # prompt_full/prompt_preview let the dashboard find tasks by prompt
        # content; NULL prompt_full rows (pre-migration) fall back to the
        # truncated preview they still carry.
        where.append(
            "(job_id LIKE ? OR prompt_id LIKE ? OR workflow LIKE ? OR error LIKE ?"
            " OR prompt_full LIKE ? OR prompt_preview LIKE ?)"
        )
        params.extend([like, like, like, like, like, like])
    if has_outputs:
        where.append("output_count > 0")
    _add_in_filter(where, params, "status", statuses)
    _add_in_filter(where, params, "kind", kinds)
    _add_in_filter(where, params, "platform", platforms)
    return where, params


def _add_in_filter(where: list[str], params: list[Any], column: str, values: list[str] | None) -> None:
    cleaned = [str(item).strip() for item in values or [] if str(item).strip()]
    if not cleaned:
        return
    where.append(f"{column} IN ({','.join('?' for _ in cleaned)})")
    params.extend(cleaned)


def _task_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["progress"] = _json_loads(item.pop("progress_json"), {})
    item["request_json"] = _json_loads(item.get("request_json"), {})
    return item


def _output_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _submission_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["params"] = _json_loads(item.pop("params_json"), {})
    task_url = str(item.pop("task_url") or "")
    item["status"] = str(item.pop("task_status") or "") or None
    item["error"] = str(item.pop("task_error") or "") or None
    if task_url:
        item["result_url"] = task_url
        primary_name = Path(task_url).name
        item["media_type"] = guess_media_type(primary_name)
    else:
        item["result_url"] = None
        item["media_type"] = None
    return item


def _json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _ensure_column(con: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _truncate(value: str, *, limit: int = 300) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
