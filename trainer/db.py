from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_db_lock = threading.Lock()


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    task TEXT,
                    target_adapter TEXT,
                    status TEXT,
                    hash TEXT,
                    assigned_to TEXT,
                    updated_at TEXT,
                    block_json TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS block_meta (
                    block_id TEXT PRIMARY KEY,
                    required_replicas INTEGER,
                    best_k INTEGER,
                    decided INTEGER,
                    created_at TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS assignments (
                    assignment_id TEXT PRIMARY KEY,
                    block_id TEXT,
                    node_id TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assignments_block_status
                ON assignments(block_id, status);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    assignment_id TEXT,
                    block_id TEXT,
                    node_id TEXT,
                    base_version INTEGER,
                    num_samples INTEGER,
                    delta_b64 TEXT,
                    metrics_json TEXT,
                    score REAL,
                    selected INTEGER,
                    created_at TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_submissions_block
                ON submissions(block_id);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_blocks_status_updated
                ON blocks(status, updated_at);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    state TEXT,
                    blocks_completed INTEGER,
                    trust_score REAL,
                    last_seen TEXT,
                    enabled INTEGER,
                    capabilities_json TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_id TEXT,
                    node_id TEXT,
                    status TEXT,
                    loss REAL,
                    updated_at TEXT,
                    model_version INTEGER
                );
                """
            )
            conn.commit()
        finally:
            conn.close()


def upsert_node(
    db_path: str,
    node_id: str,
    state: str,
    last_seen: str,
    enabled: bool = True,
    capabilities: Optional[Dict[str, Any]] = None,
) -> None:
    caps = json.dumps(capabilities or {}, sort_keys=True)
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO nodes(node_id, state, blocks_completed, trust_score, last_seen, enabled, capabilities_json)
                VALUES(?, ?, COALESCE((SELECT blocks_completed FROM nodes WHERE node_id=?), 0),
                          COALESCE((SELECT trust_score FROM nodes WHERE node_id=?), 1.0),
                          ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    state=excluded.state,
                    last_seen=excluded.last_seen,
                    enabled=excluded.enabled,
                    capabilities_json=excluded.capabilities_json;
                """,
                (node_id, state, node_id, node_id, last_seen, 1 if enabled else 0, caps),
            )
            conn.commit()
        finally:
            conn.close()


def increment_node_blocks_completed(db_path: str, node_id: str, amount: int = 1) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "UPDATE nodes SET blocks_completed=COALESCE(blocks_completed, 0) + ? WHERE node_id=?",
                (int(amount), node_id),
            )
            conn.commit()
        finally:
            conn.close()


def set_node_enabled(db_path: str, node_id: str, enabled: bool, updated_at: str) -> bool:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT node_id FROM nodes WHERE node_id=?", (node_id,))
            if cur.fetchone() is None:
                return False
            conn.execute(
                "UPDATE nodes SET enabled=?, last_seen=?, state=? WHERE node_id=?",
                (1 if enabled else 0, updated_at, "idle" if enabled else "disabled", node_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def get_node(db_path: str, node_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def list_nodes(db_path: str) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT * FROM nodes ORDER BY last_seen DESC")
            out = []
            for row in cur.fetchall():
                d = dict(row)
                try:
                    d["capabilities"] = json.loads(d.get("capabilities_json") or "{}")
                except Exception:
                    d["capabilities"] = {}
                d.pop("capabilities_json", None)
                d["enabled"] = bool(d.get("enabled"))
                out.append(d)
            return out
        finally:
            conn.close()


def enqueue_blocks(
    db_path: str,
    blocks: List[Dict[str, Any]],
    dataset_label: str,
    target_adapter: Optional[str],
    updated_at: str,
    required_replicas: int = 1,
    best_k: int = 1,
) -> int:
    added = 0
    with _db_lock:
        conn = _connect(db_path)
        try:
            for obj in blocks:
                block_id = str(obj.get("block_id") or "")
                if not block_id:
                    continue
                block_hash = obj.get("block_hash") or obj.get("hash")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO blocks(block_id, task, target_adapter, status, hash, assigned_to, updated_at, block_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        block_id,
                        dataset_label,
                        target_adapter,
                        "pending",
                        block_hash,
                        None,
                        updated_at,
                        json.dumps(obj),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO block_meta(block_id, required_replicas, best_k, decided, created_at)
                    VALUES(?, ?, ?, COALESCE((SELECT decided FROM block_meta WHERE block_id=?), 0),
                           COALESCE((SELECT created_at FROM block_meta WHERE block_id=?), ?))
                    """,
                    (block_id, int(required_replicas), int(best_k), block_id, block_id, updated_at),
                )
                added += 1
            conn.commit()
        finally:
            conn.close()
    return added


def queue_counts(db_path: str) -> Dict[str, int]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            def count(status: str) -> int:
                cur = conn.execute("SELECT COUNT(*) AS c FROM blocks WHERE status=?", (status,))
                return int(cur.fetchone()["c"])

            total = int(conn.execute("SELECT COUNT(*) AS c FROM blocks").fetchone()["c"])
            return {
                "total": total,
                "pending": count("pending"),
                "assigned": count("assigned"),
                "completed": count("completed"),
                "failed": count("failed"),
            }
        finally:
            conn.close()


def list_recent_blocks(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                """
                SELECT
                    b.block_id, b.task, b.target_adapter, b.status, b.hash, b.assigned_to, b.updated_at,
                    COALESCE(m.required_replicas, 1) AS required_replicas,
                    COALESCE(m.best_k, 1) AS best_k,
                    COALESCE(m.decided, 0) AS decided,
                    (SELECT COUNT(*) FROM submissions s WHERE s.block_id=b.block_id) AS submissions_count
                FROM blocks b
                LEFT JOIN block_meta m ON m.block_id=b.block_id
                ORDER BY b.updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def claim_next_block(db_path: str, node_id: str, updated_at: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute(
                """
                SELECT b.block_id
                FROM blocks b
                LEFT JOIN block_meta m ON m.block_id=b.block_id
                WHERE
                    COALESCE(m.decided, 0) = 0
                    AND b.status IN ('pending', 'assigned')
                    AND (
                        (SELECT COUNT(*) FROM assignments a WHERE a.block_id=b.block_id AND a.status IN ('assigned', 'completed'))
                        < COALESCE(m.required_replicas, 1)
                    )
                ORDER BY b.updated_at ASC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                conn.execute("COMMIT;")
                return None
            block_id = row["block_id"]
            assignment_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO assignments(assignment_id, block_id, node_id, status, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (assignment_id, block_id, node_id, "assigned", updated_at, updated_at),
            )
            # Mark the block as assigned/in-progress.
            conn.execute(
                "UPDATE blocks SET status='assigned', assigned_to=?, updated_at=? WHERE block_id=?",
                (node_id, updated_at, block_id),
            )
            cur3 = conn.execute(
                """
                SELECT b.block_id, b.task, b.target_adapter, b.status, b.hash, b.assigned_to, b.updated_at,
                       COALESCE(m.required_replicas, 1) AS required_replicas,
                       COALESCE(m.best_k, 1) AS best_k,
                       COALESCE(m.decided, 0) AS decided
                FROM blocks b
                LEFT JOIN block_meta m ON m.block_id=b.block_id
                WHERE b.block_id=?
                """,
                (block_id,),
            )
            conn.execute("COMMIT;")
            rec = cur3.fetchone()
            if not rec:
                return None
            d = dict(rec)
            d["assignment_id"] = assignment_id
            # include current submission count for convenience
            cur4 = conn.execute("SELECT COUNT(*) AS c FROM submissions WHERE block_id=?", (block_id,))
            d["submissions_count"] = int(cur4.fetchone()["c"])
            return d
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()


def get_assignment(db_path: str, assignment_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def mark_assignment_status(db_path: str, assignment_id: str, status: str, updated_at: str) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "UPDATE assignments SET status=?, updated_at=? WHERE assignment_id=?",
                (status, updated_at, assignment_id),
            )
            conn.commit()
        finally:
            conn.close()


def record_submission(
    db_path: str,
    assignment_id: str,
    block_id: str,
    node_id: str,
    base_version: int,
    num_samples: int,
    delta_b64: str,
    metrics: Dict[str, Any],
    score: float,
    selected: bool,
    created_at: str,
) -> str:
    submission_id = str(uuid.uuid4())
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO submissions(
                    submission_id, assignment_id, block_id, node_id, base_version, num_samples, delta_b64,
                    metrics_json, score, selected, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission_id,
                    assignment_id,
                    block_id,
                    node_id,
                    int(base_version),
                    int(num_samples),
                    delta_b64,
                    json.dumps(metrics or {}, sort_keys=True),
                    float(score),
                    1 if selected else 0,
                    created_at,
                ),
            )
            conn.commit()
            return submission_id
        finally:
            conn.close()


def block_submission_count(db_path: str, block_id: str) -> int:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT COUNT(*) AS c FROM submissions WHERE block_id=?", (block_id,))
            return int(cur.fetchone()["c"])
        finally:
            conn.close()


def get_block_meta(db_path: str, block_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT * FROM block_meta WHERE block_id=?", (block_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def set_block_decided(db_path: str, block_id: str, decided: bool, status: str, updated_at: str) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute("UPDATE block_meta SET decided=? WHERE block_id=?", (1 if decided else 0, block_id))
            conn.execute("UPDATE blocks SET status=?, updated_at=? WHERE block_id=?", (status, updated_at, block_id))
            conn.commit()
        finally:
            conn.close()


def list_block_submissions(db_path: str, block_id: str) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "SELECT submission_id, assignment_id, block_id, node_id, base_version, num_samples, delta_b64, metrics_json, score, selected, created_at "
                "FROM submissions WHERE block_id=?",
                (block_id,),
            )
            out = []
            for r in cur.fetchall():
                d = dict(r)
                try:
                    d["metrics"] = json.loads(d.get("metrics_json") or "{}")
                except Exception:
                    d["metrics"] = {}
                d.pop("metrics_json", None)
                d["selected"] = bool(d.get("selected"))
                out.append(d)
            return out
        finally:
            conn.close()


def set_selected_submissions(db_path: str, submission_ids: List[str], selected: bool) -> None:
    if not submission_ids:
        return
    with _db_lock:
        conn = _connect(db_path)
        try:
            q = ",".join(["?"] * len(submission_ids))
            conn.execute(f"UPDATE submissions SET selected=? WHERE submission_id IN ({q})", (1 if selected else 0, *submission_ids))
            conn.commit()
        finally:
            conn.close()


def get_block(db_path: str, block_id: str) -> Optional[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("SELECT block_json FROM blocks WHERE block_id=?", (block_id,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                return json.loads(row["block_json"])
            except Exception:
                return None
        finally:
            conn.close()


def mark_block_status(db_path: str, block_id: str, status: str, updated_at: str) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "UPDATE blocks SET status=?, updated_at=? WHERE block_id=?",
                (status, updated_at, block_id),
            )
            conn.commit()
        finally:
            conn.close()


def append_history(
    db_path: str,
    block_id: str,
    node_id: str,
    status: str,
    loss: Optional[float],
    updated_at: str,
    model_version: int,
) -> None:
    with _db_lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT INTO history(block_id, node_id, status, loss, updated_at, model_version) VALUES(?, ?, ?, ?, ?, ?)",
                (block_id, node_id, status, loss, updated_at, model_version),
            )
            conn.commit()
        finally:
            conn.close()


def recent_history(db_path: str, limit: int = 10) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "SELECT block_id, node_id, status, loss, updated_at, model_version FROM history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
