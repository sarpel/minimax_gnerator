"""Progress Tracker

Rich console-based progress tracking for generation sessions.
Provides real-time updates, task status, and overall progress.

Features:
- Rich console output with progress bars
- Task-level status tracking
- Overall progress monitoring
- Color-coded status indicators
- Real-time updates without flickering
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.status import Status

@dataclass
class ProgressConfig:
    """Configuration for progress tracking.

    Attributes:
        refresh_rate: Refresh rate for progress updates (seconds)
        show_task_details: Whether to show detailed task information
        console_width: Width of the console display
    """
    refresh_rate: float = 0.1
    show_task_details: bool = True
    console_width: int = 80

class ProgressTracker:
    """Rich console-based progress tracker for generation sessions.

    This class provides:
    - Real-time progress bars and status updates
    - Task-level status tracking
    - Color-coded indicators for different states
    - Memory-efficient updates
    - Clean console output
    """

    def __init__(self, config: Optional[ProgressConfig] = None):
        """Initialize the progress tracker.

        Args:
            config: Progress configuration (optional)
        """
        self.config = config or ProgressConfig()
        self.console = Console()
        self.progress = Progress()
        self.live = Live(auto_refresh=False)
        self._task_table = Table(show_header=True, expand=True)
        self._setup_ui()

        # Task tracking
        self._task_statuses: Dict[str, str] = {}
        self._task_times: Dict[str, float] = {}
        self._overall_progress = 0.0
        self._total_tasks = 0
        self._completed_tasks = 0
        self._start_time = time.time()

        # Progress bar task
        self._progress_task: Optional[TaskID] = None

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        # Configure task table
        self._task_table.add_column("Task ID", width=12)
        self._task_table.add_column("Status", width=15)
        self._task_table.add_column("Duration", width=10)
        self._task_table.add_column("Details", ratio=1)

    async def initialize_batch(self, total_tasks: int) -> None:
        """Initialize progress tracking for a new batch.

        Args:
            total_tasks: Total number of tasks in this batch
        """
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._overall_progress = 0.0
        self._start_time = time.time()
        self._task_statuses.clear()
        self._task_times.clear()

        # Add overall progress bar
        self._progress_task = self.progress.add_task(
            "[green]Overall Progress",
            total=total_tasks
        )

        # Start live display
        self.live.start(refresh_per_second=1/self.config.refresh_rate)

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        details: Optional[str] = None
    ) -> None:
        """Update the status of a specific task.

        Args:
            task_id: Task identifier
            status: New status (pending, processing, completed, failed, timeout, provider_error)
            details: Additional details about the task
        """
        # Record status and time
        self._task_statuses[task_id] = status

        if status not in ['pending', 'processing']:
            # Task completed, record duration
            if task_id in self._task_times:
                duration = time.time() - self._task_times[task_id]
                self._task_times[task_id] = duration
            else:
                self._task_times[task_id] = 0.0

        if status == 'processing' and task_id not in self._task_times:
            # Task started, record start time
            self._task_times[task_id] = time.time()

        # Update progress
        if status == 'completed':
            self._completed_tasks += 1
            self._overall_progress = self._completed_tasks / self._total_tasks if self._total_tasks > 0 else 0.0

            if self._progress_task:
                self.progress.update(self._progress_task, completed=self._completed_tasks)

        # Update display
        await self._update_display()

    async def update_overall_progress(self, completed: int, total: int) -> None:
        """Update the overall progress.

        Args:
            completed: Number of completed tasks
            total: Total number of tasks
        """
        self._completed_tasks = completed
        self._total_tasks = total
        self._overall_progress = completed / total if total > 0 else 0.0

        if self._progress_task:
            self.progress.update(self._progress_task, completed=completed, total=total)

        await self._update_display()

    async def _update_display(self) -> None:
        """Update the console display with current progress."""
        # Create status panel
        elapsed_time = time.time() - self._start_time
        eta = (elapsed_time / self._completed_tasks * (self._total_tasks - self._completed_tasks)) if self._completed_tasks > 0 else 0

        status_panel = Panel(
            f"[bold]Generation Progress[/bold]\n"
            f"Completed: {self._completed_tasks}/{self._total_tasks} "
            f"({self._overall_progress:.1%})\n"
            f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s",
            title="Status",
            border_style="blue"
        )

        # Update task table
        self._task_table.rows.clear()

        # Show recent tasks (limit to 10 for performance)
        recent_tasks = list(self._task_statuses.items())[-10:]

        for task_id, status in recent_tasks:
            # Get status color and icon
            status_color = "green" if status == "completed" else "red" if status == "failed" else "yellow"
            status_icon = "✓" if status == "completed" else "✗" if status == "failed" else "⏳"

            # Get duration
            duration = self._task_times.get(task_id, 0.0)

            # Add row to table
            self._task_table.add_row(
                task_id[:12],  # Shorten task ID
                f"[{status_color}]{status_icon} {status}[/{status_color}]",
                f"{duration:.1f}s",
                str(details) if (details := None) else ""  # Placeholder for details
            )

        # Combine all elements
        display_content = "\n".join([
            str(self.progress),
            str(status_panel),
            str(self._task_table)
        ])

        # Update live display
        self.live.update(display_content)
        self.live.refresh()

    async def finalize_batch(self) -> None:
        """Finalize progress tracking for the current batch."""
        if hasattr(self, '_progress_task') and self._progress_task:
            self.progress.update(self._progress_task, completed=self._total_tasks)

        # Show final summary
        elapsed_time = time.time() - self._start_time
        success_rate = (self._completed_tasks / self._total_tasks * 100) if self._total_tasks > 0 else 0

        final_panel = Panel(
            f"[bold green]Generation Complete![/bold green]\n"
            f"Total Tasks: {self._total_tasks}\n"
            f"Completed: {self._completed_tasks}\n"
            f"Success Rate: {success_rate:.1f}%\n"
            f"Total Time: {elapsed_time:.1f}s\n"
            f"Average Task Time: {(elapsed_time / self._total_tasks):.2f}s" if self._total_tasks > 0 else "N/A",
            title="Summary",
            border_style="green"
        )

        self.live.update(str(final_panel))
        self.live.refresh()

        # Stop live display
        self.live.stop()

    async def show_error(self, error_message: str) -> None:
        """Show an error message in the progress display.

        Args:
            error_message: Error message to display
        """
        error_panel = Panel(
            f"[bold red]Error[/bold red]\n{error_message}",
            title="Generation Error",
            border_style="red"
        )

        if hasattr(self, 'live') and self.live:
            self.live.update(str(error_panel))
            self.live.refresh()
        else:
            self.console.print(error_panel)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, 'live') and self.live:
            self.live.stop()

    def get_current_status(self) -> Dict[str, Any]:
        """Get the current progress status.

        Returns:
            Dictionary with current progress information
        """
        return {
            "completed_tasks": self._completed_tasks,
            "total_tasks": self._total_tasks,
            "progress": self._overall_progress,
            "elapsed_time": time.time() - self._start_time,
            "task_statuses": dict(self._task_statuses)
        }