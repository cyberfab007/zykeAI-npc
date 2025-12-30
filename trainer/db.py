from __future__ import annotations

import json
import sqlite3
import threading
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
                "SELECT block_id, task, target_adapter, status, hash, assigned_to, updated_at FROM blocks ORDER BY updated_at DESC LIMIT ?",
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
                "SELECT block_id FROM blocks WHERE status='pending' ORDER BY updated_at ASC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                conn.execute("COMMIT;")
                return None
            block_id = row["block_id"]
            cur2 = conn.execute(
                "UPDATE blocks SET status='assigned', assigned_to=?, updated_at=? WHERE block_id=? AND status='pending'",
                (node_id, updated_at, block_id),
            )
            if cur2.rowcount != 1:
                conn.execute("COMMIT;")
                return None
            cur3 = conn.execute(
                "SELECT block_id, task, target_adapter, status, hash, assigned_to, updated_at FROM blocks WHERE block_id=?",
                (block_id,),
            )
            conn.execute("COMMIT;")
            rec = cur3.fetchone()
            return dict(rec) if rec else None
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
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
