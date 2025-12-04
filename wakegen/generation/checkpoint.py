"""Checkpoint System

SQLite-based checkpoint system for saving and resuming generation progress.
Allows crash recovery and continuation of interrupted generation sessions.

Features:
- SQLite database for persistent storage
- Task state tracking (pending, completed, failed)
- Progress snapshot and restore
- Atomic operations for data integrity
- Cleanup of old checkpoints
"""

from __future__ import annotations

import asyncio
import aiosqlite
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from wakegen.models.generation import GenerationParameters, GenerationResult
from wakegen.core.exceptions import GenerationError

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint system.

    Attributes:
        db_path: Path to SQLite database file
        cleanup_interval: Interval for cleaning up old checkpoints (seconds)
        max_checkpoints: Maximum number of checkpoints to keep
    """
    db_path: str = "checkpoints.db"
    cleanup_interval: int = 3600  # 1 hour
    max_checkpoints: int = 10

class CheckpointManager:
    """SQLite-based checkpoint manager for generation progress.

    This class handles:
    - Creating and managing checkpoint databases
    - Saving generation progress and task states
    - Restoring from checkpoints after crashes
    - Cleaning up old checkpoints
    - Atomic operations for data integrity
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize the checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self.config = config
        self._db: Optional[aiosqlite.Connection] = None
        self._last_cleanup: float = 0

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection.

        Returns:
            SQLite database connection
        """
        if self._db is None:
            # Create directory if it doesn't exist
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self._db = await aiosqlite.connect(self.config.db_path)

            # Initialize database schema
            await self._initialize_schema()

        return self._db

    async def _initialize_schema(self) -> None:
        """Initialize database schema if it doesn't exist."""
        db = await self._get_connection()

        # Create checkpoints table
        await db.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL,
            total_tasks INTEGER NOT NULL,
            completed_tasks INTEGER NOT NULL,
            config_json TEXT NOT NULL
        )
        """)

        # Create tasks table
        await db.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL,
            parameters_json TEXT,
            result_json TEXT,
            error_message TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
        )
        """)

        # Create index for faster lookups
        await db.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON checkpoints(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_checkpoint ON tasks(checkpoint_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")

        await db.commit()

    async def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints to prevent database bloat."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.config.cleanup_interval:
            return

        try:
            db = await self._get_connection()

            # Get all checkpoint IDs ordered by updated_at
            cursor = await db.execute("""
                SELECT id FROM checkpoints
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET ?
            """, (self.config.max_checkpoints,))

            old_checkpoint_ids = [row[0] async for row in cursor]

            if old_checkpoint_ids:
                # Delete old checkpoints and their tasks
                for checkpoint_id in old_checkpoint_ids:
                    await db.execute("DELETE FROM tasks WHERE checkpoint_id = ?", (checkpoint_id,))
                    await db.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))

                await db.commit()
                self._last_cleanup = current_time

        except Exception as e:
            # Don't fail if cleanup fails
            pass

    async def create_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        total_tasks: int,
        config: Dict[str, Any]
    ) -> None:
        """Create a new checkpoint for a generation session.

        Args:
            session_id: Unique session identifier
            checkpoint_id: Unique checkpoint identifier
            total_tasks: Total number of tasks in this session
            config: Configuration data to store
        """
        await self._cleanup_old_checkpoints()

        db = await self._get_connection()
        current_time = int(time.time())

        await db.execute("""
            INSERT INTO checkpoints
            (id, session_id, created_at, updated_at, status, progress, total_tasks, completed_tasks, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint_id,
            session_id,
            current_time,
            current_time,
            "active",
            0.0,
            total_tasks,
            0,
            json.dumps(config)
        ))

        await db.commit()

    async def save_task_state(
        self,
        checkpoint_id: str,
        task_id: str,
        status: str,
        parameters: Optional[GenerationParameters] = None,
        result: Optional[GenerationResult] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Save the state of a single task.

        Args:
            checkpoint_id: Checkpoint identifier
            task_id: Task identifier
            status: Task status (pending, processing, completed, failed)
            parameters: Generation parameters (optional)
            result: Generation result (optional)
            error: Error information (optional)
        """
        db = await self._get_connection()
        current_time = int(time.time())

        parameters_json = json.dumps(parameters.dict()) if parameters else None
        result_json = json.dumps(result.dict()) if result else None
        error_message = str(error) if error else None

        await db.execute("""
            INSERT OR REPLACE INTO tasks
            (id, checkpoint_id, task_id, status, parameters_json, result_json, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{checkpoint_id}_{task_id}",
            checkpoint_id,
            task_id,
            status,
            parameters_json,
            result_json,
            error_message,
            current_time,
            current_time
        ))

        # Update checkpoint progress
        if status == "completed":
            cursor = await db.execute("""
                SELECT completed_tasks FROM checkpoints WHERE id = ?
            """, (checkpoint_id,))
            row = await cursor.fetchone()
            if row:
                completed_tasks = row[0] + 1
                total_tasks = await self._get_total_tasks(checkpoint_id)
                progress = completed_tasks / total_tasks if total_tasks > 0 else 0.0

                await db.execute("""
                    UPDATE checkpoints
                    SET completed_tasks = ?, progress = ?, updated_at = ?
                    WHERE id = ?
                """, (completed_tasks, progress, current_time, checkpoint_id))

        await db.commit()

    async def _get_total_tasks(self, checkpoint_id: str) -> int:
        """Get total number of tasks for a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Total number of tasks
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT total_tasks FROM checkpoints WHERE id = ?
        """, (checkpoint_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_checkpoint_status(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get the status of a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Dictionary with checkpoint status information
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT status, progress, total_tasks, completed_tasks, config_json
            FROM checkpoints
            WHERE id = ?
        """, (checkpoint_id,))

        row = await cursor.fetchone()
        if not row:
            raise GenerationError(f"Checkpoint {checkpoint_id} not found")

        return {
            "status": row[0],
            "progress": row[1],
            "total_tasks": row[2],
            "completed_tasks": row[3],
            "config": json.loads(row[4])
        }

    async def get_pending_tasks(self, checkpoint_id: str) -> List[Tuple[str, GenerationParameters]]:
        """Get all pending tasks for a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            List of (task_id, parameters) tuples for pending tasks
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT task_id, parameters_json
            FROM tasks
            WHERE checkpoint_id = ? AND status IN ('pending', 'processing')
        """, (checkpoint_id,))

        tasks = []
        async for row in cursor:
            task_id = row[0]
            parameters_json = row[1]
            if parameters_json:
                parameters = GenerationParameters.parse_raw(parameters_json)
                tasks.append((task_id, parameters))

        return tasks

    async def get_completed_tasks(self, checkpoint_id: str) -> List[Tuple[str, GenerationResult]]:
        """Get all completed tasks for a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            List of (task_id, result) tuples for completed tasks
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT task_id, result_json
            FROM tasks
            WHERE checkpoint_id = ? AND status = 'completed'
        """, (checkpoint_id,))

        tasks = []
        async for row in cursor:
            task_id = row[0]
            result_json = row[1]
            if result_json:
                result = GenerationResult.parse_raw(result_json)
                tasks.append((task_id, result))

        return tasks

    async def get_failed_tasks(self, checkpoint_id: str) -> List[Tuple[str, str]]:
        """Get all failed tasks for a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            List of (task_id, error_message) tuples for failed tasks
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT task_id, error_message
            FROM tasks
            WHERE checkpoint_id = ? AND status = 'failed'
        """, (checkpoint_id,))

        return [row async for row in cursor]

    async def mark_checkpoint_completed(self, checkpoint_id: str) -> None:
        """Mark a checkpoint as completed.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        db = await self._get_connection()
        current_time = int(time.time())

        await db.execute("""
            UPDATE checkpoints
            SET status = 'completed', updated_at = ?
            WHERE id = ?
        """, (current_time, checkpoint_id))

        await db.commit()

    async def mark_checkpoint_failed(self, checkpoint_id: str, error_message: str) -> None:
        """Mark a checkpoint as failed.

        Args:
            checkpoint_id: Checkpoint identifier
            error_message: Error message describing the failure
        """
        db = await self._get_connection()
        current_time = int(time.time())

        await db.execute("""
            UPDATE checkpoints
            SET status = 'failed', updated_at = ?
            WHERE id = ?
        """, (current_time, checkpoint_id))

        await db.commit()

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint and all its associated tasks.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        db = await self._get_connection()

        # Delete tasks first (foreign key constraint)
        await db.execute("DELETE FROM tasks WHERE checkpoint_id = ?", (checkpoint_id,))

        # Delete checkpoint
        await db.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))

        await db.commit()

    async def get_latest_checkpoint(self, session_id: str) -> Optional[str]:
        """Get the latest checkpoint ID for a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest checkpoint ID or None if not found
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT id FROM checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (session_id,))

        row = await cursor.fetchone()
        return row[0] if row else None

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Tuple[Dict[str, Any], List[Tuple[str, GenerationParameters]]]:
        """Restore generation state from a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Tuple of (config, pending_tasks)
        """
        # Get checkpoint status
        status = await self.get_checkpoint_status(checkpoint_id)

        # Get pending tasks
        pending_tasks = await self.get_pending_tasks(checkpoint_id)

        return status["config"], pending_tasks

    async def save_batch_progress(
        self,
        checkpoint_id: str,
        task_results: List[Tuple[str, Optional[GenerationResult], Optional[Exception]]]
    ) -> None:
        """Save progress for a batch of tasks.

        Args:
            checkpoint_id: Checkpoint identifier
            task_results: List of (task_id, result, error) tuples
        """
        db = await self._get_connection()

        for task_id, result, error in task_results:
            status = "completed" if result else "failed"
            await self.save_task_state(
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                status=status,
                result=result,
                error=error
            )