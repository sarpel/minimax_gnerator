# ADR-002: SQLite Checkpoints for Generation Jobs

**Status:** Accepted  
**Date:** 2024-12-13

## Context

Generating wake word datasets can take hours for large sample counts. If the process is interrupted (crash, power failure, user abort), we need to resume without losing progress.

Options considered:
1. **In-memory tracking** - Store progress in Python data structures
2. **SQLite database** - Persistent local database
3. **Redis** - In-memory data store with persistence
4. **JSON files** - Simple file-based state

## Decision

We chose **SQLite with async support (aiosqlite)** for checkpoint persistence.

```python
# Checkpoint schema
CREATE TABLE generation_checkpoints (
    job_id TEXT PRIMARY KEY,
    status TEXT,
    completed_samples INTEGER,
    total_samples INTEGER,
    last_updated TIMESTAMP,
    metadata JSON
);

CREATE TABLE sample_status (
    sample_id TEXT PRIMARY KEY,
    job_id TEXT,
    status TEXT,  -- pending, completed, failed
    output_path TEXT,
    error_message TEXT,
    FOREIGN KEY (job_id) REFERENCES generation_checkpoints(job_id)
);
```

## Rationale

### Why SQLite?

1. **Zero Configuration**: No server to install or manage
2. **ACID Compliance**: Guaranteed data integrity on crash
3. **Async Support**: `aiosqlite` integrates with our async architecture
4. **Query Capability**: Can query job status, filter samples, aggregate stats
5. **Single File**: Easy to backup, delete, or move

### Why NOT In-Memory?

- Lost on crash or process termination
- No resume capability

### Why NOT Redis?

- Requires separate server process
- Overkill for single-user local application
- Memory constraints for large datasets

### Why NOT JSON Files?

- No ACID guarantees (corruption on crash)
- Poor query performance
- Concurrent write issues

## Consequences

### Positive
- Reliable crash recovery
- Can resume interrupted jobs
- Progress visible to web UI via database queries

### Negative
- Slight overhead for database operations
- Requires cleanup of old checkpoint data
- File locking on Windows can be tricky

## Related
- Audio files stored on filesystem (see ADR-001)
- Web UI reads checkpoint data for progress updates
