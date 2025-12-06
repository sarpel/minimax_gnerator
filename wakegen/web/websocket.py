"""
WebSocket Handler for Real-Time Progress Updates

This module provides WebSocket endpoints for streaming real-time updates
during audio generation. Instead of polling the /status endpoint,
clients can connect to a WebSocket and receive push notifications.

    WHY WEBSOCKETS?
    ===============
    HTTP is request-response: client asks, server answers.
    WebSocket is bidirectional: server can push data to client anytime.

    For generation progress, this means:
    - No polling delay (instant updates)
    - Less network overhead (no repeated HTTP headers)
    - Better user experience (smooth progress bar)

    HOW IT WORKS:
    =============
    1. Client connects to /ws/progress/{job_id}
    2. Server sends JSON messages as progress updates
    3. Messages include: status, percentage, current file, etc.
    4. Connection closes when job completes or fails

    MESSAGE FORMAT:
    ===============
    {
        "type": "progress",
        "job_id": "abc123",
        "status": "running",
        "progress_percentage": 45.5,
        "completed_samples": 23,
        "total_samples": 50,
        "current_word": "hey assistant",
        "current_file": "sample_0023.wav"
    }
"""

import asyncio
import logging
from typing import Dict, Set, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Import job management from generation router
from wakegen.web.routers.generation import get_job, JobStatus

logger = logging.getLogger(__name__)


# =============================================================================
# CONNECTION MANAGER
# =============================================================================
# Manages all active WebSocket connections, grouped by job_id.
# This allows us to broadcast updates to all clients watching a specific job.


class ConnectionManager:
    """
    Manages WebSocket connections for progress updates.

    This is a singleton-style manager that tracks all active connections.
    When a job progresses, we can notify all clients watching that job.

        PATTERN EXPLANATION:
        ====================
        We use a dictionary where:
        - Key = job_id (string)
        - Value = set of WebSocket connections

        A set is used because:
        - Fast add/remove operations
        - No duplicates
        - We don't care about order

        THREAD SAFETY:
        ==============
        FastAPI runs in an async event loop, so we don't need traditional
        thread locks. Async operations are cooperative - only one runs at a time.
    """

    def __init__(self) -> None:
        """Initialize the connection manager with empty tracking dictionaries."""
        # Maps job_id -> set of active WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # Track when each connection was established (for debugging/timeouts)
        self.connection_times: Dict[WebSocket, datetime] = {}

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Accept a new WebSocket connection and start tracking it.

        Args:
            websocket: The WebSocket connection to accept
            job_id: The job this client wants to watch

        WHAT HAPPENS:
        =============
        1. Accept the WebSocket handshake
        2. Add to the set of connections for this job_id
        3. Record connection time for debugging
        """
        # Accept the WebSocket handshake
        # This completes the HTTP -> WebSocket upgrade
        await websocket.accept()

        # Create a set for this job if it doesn't exist
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()

        # Add this connection to the job's watchers
        self.active_connections[job_id].add(websocket)
        self.connection_times[websocket] = datetime.now()

        logger.info(f"WebSocket connected for job {job_id}")

    def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Remove a WebSocket connection when it closes.

        Args:
            websocket: The connection to remove
            job_id: The job this connection was watching
        """
        # Remove from the job's connection set
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

        # Remove from connection times
        self.connection_times.pop(websocket, None)

        logger.info(f"WebSocket disconnected for job {job_id}")

    async def send_progress(self, job_id: str, data: dict) -> None:
        """
        Send a progress update to all clients watching a job.

        Args:
            job_id: The job to send updates for
            data: The progress data to send (will be JSON-serialized)

        NOTE ON ERROR HANDLING:
        =======================
        If a send fails, the connection is probably dead.
        We catch the error and disconnect that client gracefully.
        """
        if job_id not in self.active_connections:
            return

        # Make a copy of the set to avoid modification during iteration
        connections = list(self.active_connections[job_id])

        for websocket in connections:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                self.disconnect(websocket, job_id)

    async def broadcast_all(self, data: dict) -> None:
        """
        Send a message to ALL connected clients (all jobs).

        Useful for system-wide notifications like server shutdown.
        """
        for job_id in list(self.active_connections.keys()):
            await self.send_progress(job_id, data)

    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        """
        Get the number of active connections.

        Args:
            job_id: If provided, count only connections for this job.
                   If None, count all connections.
        """
        if job_id:
            return len(self.active_connections.get(job_id, set()))

        return sum(len(conns) for conns in self.active_connections.values())


# Global connection manager instance
# This is shared across all WebSocket endpoints
manager = ConnectionManager()


# =============================================================================
# WEBSOCKET ROUTER
# =============================================================================


router = APIRouter()


@router.websocket("/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint for real-time job progress updates.

    Clients connect to this endpoint to receive live updates about a
    generation job's progress without polling.

        PATH PARAMETERS:
        ================
        job_id: The ID of the job to watch (from POST /api/generate/start)

        MESSAGE TYPES SENT:
        ===================
        - "connected": Initial connection confirmation
        - "progress": Regular progress updates
        - "completed": Job finished successfully
        - "failed": Job encountered an error
        - "cancelled": Job was cancelled

        EXAMPLE CLIENT CODE (JavaScript):
        =================================
        const ws = new WebSocket('ws://localhost:8080/ws/progress/abc123');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.progress_percentage);
        };
    """
    # Accept the connection and register it
    await manager.connect(websocket, job_id)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "Connected to progress stream"
        })

        # Check if job exists
        job = get_job(job_id)
        if not job:
            await websocket.send_json({
                "type": "error",
                "message": f"Job not found: {job_id}"
            })
            return

        # Send current status immediately
        await websocket.send_json({
            "type": "progress",
            "job_id": job.id,
            "status": job.status.value,
            "progress_percentage": job.progress_percentage,
            "completed_samples": job.completed_samples,
            "total_samples": job.total_samples,
            "current_word": job.current_word,
            "current_file": job.current_file
        })

        # Keep connection alive and send updates
        # We poll the job status and push updates
        while True:
            # Wait a bit before checking again
            await asyncio.sleep(0.5)

            # Get latest job status
            job = get_job(job_id)
            if not job:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job no longer exists"
                })
                break

            # Send progress update
            await websocket.send_json({
                "type": "progress",
                "job_id": job.id,
                "status": job.status.value,
                "progress_percentage": job.progress_percentage,
                "completed_samples": job.completed_samples,
                "total_samples": job.total_samples,
                "current_word": job.current_word,
                "current_file": job.current_file
            })

            # Check if job is done
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                # Send final status
                await websocket.send_json({
                    "type": job.status.value,
                    "job_id": job.id,
                    "message": f"Job {job.status.value}",
                    "completed_samples": job.completed_samples,
                    "error_message": job.error_message
                })
                break

    except WebSocketDisconnect:
        # Client disconnected normally
        logger.info(f"Client disconnected from job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass

    finally:
        # Always clean up the connection
        manager.disconnect(websocket, job_id)


@router.websocket("/stats")
async def websocket_stats(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time system statistics.

    Broadcasts system stats like active jobs, provider status, etc.
    Useful for dashboard live updates.
    """
    await websocket.accept()

    try:
        while True:
            # Send stats every 5 seconds
            await asyncio.sleep(5)

            await websocket.send_json({
                "type": "stats",
                "active_connections": manager.get_connection_count(),
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def notify_job_progress(job_id: str, progress_data: dict) -> None:
    """
    Utility function to notify all WebSocket clients about job progress.

    Called from the generation runner when progress changes.

    Args:
        job_id: The job that made progress
        progress_data: Dictionary with progress info
    """
    progress_data["type"] = "progress"
    progress_data["job_id"] = job_id
    await manager.send_progress(job_id, progress_data)


def get_connection_manager() -> ConnectionManager:
    """
    Get the global connection manager.

    Use this to access the manager from other modules.
    """
    return manager
