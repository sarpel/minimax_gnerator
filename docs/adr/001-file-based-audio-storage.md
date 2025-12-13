# ADR-001: File-Based Audio Storage

**Status:** Accepted  
**Date:** 2024-12-13

## Context

WakeGen generates thousands of audio samples during dataset creation. We needed to decide how to store these audio files:

1. **File System** - Store `.wav` files directly on disk
2. **Database BLOBs** - Store audio data in SQLite or PostgreSQL
3. **Object Storage** - Use S3/MinIO for cloud-compatible storage

## Decision

We chose **file system storage** with the following structure:

```
output/
├── raw/                    # Original TTS outputs
│   ├── hey_assistant/
│   │   ├── sample_001.wav
│   │   └── sample_002.wav
│   └── ok_computer/
├── augmented/              # Post-augmentation samples
└── exports/                # Final dataset exports
```

## Rationale

### Why File System?

1. **Simplicity**: No additional dependencies or database setup
2. **Performance**: Direct file I/O is faster for large binary files
3. **Tooling**: Easy to inspect, play, and debug with standard audio tools
4. **Portability**: Users can easily copy/move datasets between systems
5. **Memory Efficiency**: Streaming file access vs. loading entire BLOBs

### Why NOT Database BLOBs?

- SQLite has poor performance with large BLOBs (>1MB)
- Increases database size significantly
- Harder to debug and inspect audio files
- No benefit for non-relational audio data

### Why NOT Object Storage?

- Adds infrastructure complexity
- Requires network for local development
- Overkill for single-machine use case

## Consequences

### Positive
- Fast and simple implementation
- Works offline without any setup
- Easy to backup and restore datasets

### Negative
- Not suitable for multi-machine distributed processing
- Manual file management required
- No built-in versioning or metadata search

## Related
- Checkpoint system uses SQLite for lightweight task metadata (see ADR-002)
