# Future Improvements & Refactoring Plan

**Status:** Planned
**Priority:** Medium/Low
**Objective:** Address remaining technical debt, architectural improvements, and non-critical code quality issues identified during the comprehensive code review. These items are scheduled for a future development cycle.

## 1. Architecture & Design

### AR-004: Circuit Breaker for TTS Providers
*   **Current State:** Failed requests to external APIs (like MiniMax) might trigger immediate retries or fail, potentially hammering a struggling service.
*   **Plan:** Implement a Circuit Breaker pattern (e.g., using `pybreaker` or custom implementation) around `BaseProvider.generate` calls.
*   **Benefit:** Improves system resilience and prevents cascading failures.

### AR-002: Dependency Injection for ConnectionManager
*   **Current State:** The WebSocket `ConnectionManager` is likely instantiated as a global singleton.
*   **Plan:** Refactor `wakegen/web/websocket.py` and `wakegen/web/app.py` to use FastAPI's dependency injection system (`Depends`) for the connection manager.
*   **Benefit:** Improves testability (easier to mock) and manages lifecycle better.

### PERF-003: Scalable Event-Driven Architecture
*   **Current State:** WebSocket updates use an internal mechanism that works for a single process.
*   **Plan:** Design a proper Pub/Sub layer (potentially using Redis) for job status updates.
*   **Benefit:** Allows the web server to scale to multiple worker processes without losing WebSocket state.

## 2. Code Quality & Cleanup

### AH-001: De-duplicate `BatchConfig`
*   **Current State:** `BatchConfig` class is defined in `wakegen/config/batch.py` and potentially redefined or shadowed in `wakegen/utils/async_helpers.py`.
*   **Plan:** Consolidate into a single definition in `wakegen/config/batch.py` and import it elsewhere.
*   **Benefit:** Reduces confusion and maintenance overhead.

### BP-003: Fix Provider Metadata Tracking
*   **Current State:** The `_current_provider_type` attribute in `BatchProcessor` might not be accurately updated during provider fallback scenarios.
*   **Plan:** Audit `BatchProcessor` and `GenerationOrchestrator` to ensure the final `AudioSample` metadata accurately reflects the *actual* provider used, even after a fallback.
*   **Benefit:** Ensures dataset metadata is accurate for training.

### OS-001: Import Organization
*   **Current State:** Some providers (e.g., `coqui_xtts.py`) have imports inside methods that could be moved to the module level or handled with `TYPE_CHECKING` blocks.
*   **Plan:** Audit all provider files and standardise import placement following PEP 8.
*   **Benefit:** Cleaner code and better static analysis.

## 3. Documentation

### Architecture Decision Records (ADRs)
*   **Plan:** Document key architectural decisions (e.g., why we chose file-based audio storage, why we use SQLite for checkpoints) in `docs/adr/`.
*   **Benefit:** Preserves context for future contributors.

### Production Deployment Guide
*   **Plan:** Create `docs/deployment.md` covering Docker setup, Nginx reverse proxy configuration, and SSL setup.
*   **Benefit:** Lowers barrier to entry for production usage.

## 4. Testing

### Coverage Expansion
*   **Plan:** Increase test coverage from current ~60% to 80%+.
    *   Add unit tests for `wakegen/utils/` (audio, async helpers).
    *   Add integration tests for `wakegen/web` error handling.
*   **Benefit:** Higher confidence in refactoring and release stability.
