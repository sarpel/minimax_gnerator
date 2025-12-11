"""Persistent Job Storage Module

AR-001 Fix: Implements SQLite-based persistent job storage to replace
in-memory storage. Jobs survive server restarts and can be recovered.

ELI5: Instead of keeping job information only in the computer's memory
(which disappears when the server restarts), we save it to a database file
on disk. This way, if the server crashes or restarts, we don't lose track
of what jobs were running.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "wakegen_jobs.db"


class JobStorage:
    """
    Persistent job storage using SQLite.

    AR-001 Fix: This class provides persistent storage for generation jobs,
    replacing the in-memory dictionary. Jobs are stored in a SQLite database
    and can be recovered after server restarts.

    ELI5: Think of this like a filing cabinet for job information. Instead of
    keeping papers on your desk (memory), we file them away in a cabinet (database)
    so they're safe even if you leave the office.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        """
        Initialize job storage with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"Initialized job storage at {self.db_path}")

    def _init_database(self) -> None:
        """
        Create database tables if they don't exist.

        AR-001 Fix: Sets up the database schema for storing jobs.
        Uses JSON for flexible storage of job configuration and progress.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    
                    -- Configuration (stored as JSON)
                    wake_words TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    
                    -- Progress
                    total_samples INTEGER DEFAULT 0,
                    completed_samples INTEGER DEFAULT 0,
                    current_word TEXT,
                    current_file TEXT,
                    error_message TEXT,
                    
                    -- Metadata
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Create index on status for faster queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status 
                ON jobs(status)
            """
            )

            # Create index on created_at for sorting
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at 
                ON jobs(created_at DESC)
            """
            )

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection with proper error handling.

        Yields:
            SQLite connection

        ELI5: This opens the filing cabinet, lets us work with it,
        and makes sure we close it properly when we're done.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def save_job(self, job_data: dict) -> None:
        """
        Save or update a job in the database.

        Args:
            job_data: Job data dictionary (from GenerationJob.model_dump())

        AR-001 Fix: Persists job to SQLite using UPSERT (insert or update).
        """
        with self._get_connection() as conn:
            # Convert wake_words list to JSON string
            wake_words_json = json.dumps(job_data["wake_words"])

            # Format datetime fields
            created_at = (
                job_data["created_at"].isoformat()
                if isinstance(job_data["created_at"], datetime)
                else job_data["created_at"]
            )
            started_at = (
                job_data["started_at"].isoformat()
                if job_data.get("started_at")
                and isinstance(job_data["started_at"], datetime)
                else job_data.get("started_at")
            )
            completed_at = (
                job_data["completed_at"].isoformat()
                if job_data.get("completed_at")
                and isinstance(job_data["completed_at"], datetime)
                else job_data.get("completed_at")
            )
            updated_at = datetime.now().isoformat()

            conn.execute(
                """
                INSERT INTO jobs (
                    id, status, created_at, started_at, completed_at,
                    wake_words, count, provider, output_dir,
                    total_samples, completed_samples, current_word, current_file,
                    error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    total_samples = excluded.total_samples,
                    completed_samples = excluded.completed_samples,
                    current_word = excluded.current_word,
                    current_file = excluded.current_file,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
            """,
                (
                    job_data["id"],
                    job_data["status"],
                    created_at,
                    started_at,
                    completed_at,
                    wake_words_json,
                    job_data["count"],
                    job_data["provider"],
                    job_data["output_dir"],
                    job_data["total_samples"],
                    job_data["completed_samples"],
                    job_data.get("current_word"),
                    job_data.get("current_file"),
                    job_data.get("error_message"),
                    updated_at,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        """
        Retrieve a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job data dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def list_jobs(self, limit: int = 10, status: str | None = None) -> list[dict]:
        """
        List recent jobs, optionally filtered by status.

        Args:
            limit: Maximum number of jobs to return
            status: Optional status filter

        Returns:
            List of job data dictionaries
        """
        with self._get_connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                )

            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from storage.

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Delete completed/failed jobs older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of jobs deleted

        AR-001 Fix: Prevents database from growing indefinitely by
        removing old completed jobs.
        """
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM jobs 
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND created_at < ?
            """,
                (cutoff_iso,),
            )
            conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old jobs")

            return deleted

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """
        Convert SQLite row to job data dictionary.

        Args:
            row: SQLite row object

        Returns:
            Job data dictionary
        """
        return {
            "id": row["id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "wake_words": json.loads(row["wake_words"]),
            "count": row["count"],
            "provider": row["provider"],
            "output_dir": row["output_dir"],
            "total_samples": row["total_samples"],
            "completed_samples": row["completed_samples"],
            "current_word": row["current_word"],
            "current_file": row["current_file"],
            "error_message": row["error_message"],
        }


# Global storage instance
_storage: JobStorage | None = None


def get_storage(db_path: str | Path = DEFAULT_DB_PATH) -> JobStorage:
    """
    Get the global job storage instance.

    Args:
        db_path: Path to database file

    Returns:
        JobStorage instance

    ELI5: This gives us access to the filing cabinet. The first time
    we call it, it creates the cabinet. After that, it just gives us
    the same cabinet we created before.
    """
    global _storage
    if _storage is None:
        _storage = JobStorage(db_path)
    return _storage
