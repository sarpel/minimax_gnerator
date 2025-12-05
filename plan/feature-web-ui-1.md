---
goal: Create a comprehensive Web UI for WakeGen that exposes all functionality of the project
version: 1.0
date_created: 2025-12-06
last_updated: 2025-12-06
owner: WakeGen Development Team
status: 'Planned'
tags: [feature, web-ui, frontend, backend, api]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This implementation plan outlines the creation of a comprehensive Web UI for WakeGen - the Wake Word Dataset Generator. The Web UI will provide a beautiful, intuitive interface for all project functionality including TTS provider management, audio generation, augmentation, quality assurance, and dataset export.

## 1. Requirements & Constraints

### Functional Requirements

- **REQ-001**: Display all 11 TTS providers with availability status, voices, and configuration options
- **REQ-002**: Allow creation and editing of YAML configurations through a visual form interface
- **REQ-003**: Support batch generation of wake word samples with real-time progress tracking
- **REQ-004**: Provide visual configuration of all 6 environment augmentation profiles
- **REQ-005**: Display all 15+ device presets with their specifications and augmentation parameters
- **REQ-006**: Support all 6 export formats (OpenWakeWord, Mycroft Precise, Picovoice, TensorFlow, PyTorch, HuggingFace)
- **REQ-007**: Provide audio playback and waveform visualization for generated samples
- **REQ-008**: Show real-time quality metrics (SNR, duration, validation status)
- **REQ-009**: Display cache statistics and provide cache management interface
- **REQ-010**: Show GPU status and availability for GPU-accelerated providers
- **REQ-011**: Support drag-and-drop file upload for voice cloning and custom RIRs
- **REQ-012**: Provide dark/light theme support

### Security Requirements

- **SEC-001**: Secure API key storage with environment variable fallback
- **SEC-002**: Input validation on all user-provided data
- **SEC-003**: CORS configuration for local development
- **SEC-004**: Rate limiting on generation endpoints

### Technical Constraints

- **CON-001**: Must integrate with existing async Python codebase
- **CON-002**: Must support WebSocket for real-time progress updates
- **CON-003**: Should work on Windows, macOS, and Linux
- **CON-004**: Must maintain CLI compatibility (UI is additive, not replacing)
- **CON-005**: Should have minimal additional dependencies

### Design Guidelines

- **GUD-001**: Use consistent color scheme matching wake word/audio theme (purple/blue gradients)
- **GUD-002**: Responsive design supporting desktop and tablet views
- **GUD-003**: Use card-based layouts for provider and preset displays
- **GUD-004**: Provide clear visual feedback for async operations
- **GUD-005**: Use iconography for provider types and status indicators

### Architecture Patterns

- **PAT-001**: Backend-for-Frontend (BFF) pattern with FastAPI
- **PAT-002**: WebSocket for real-time bidirectional communication
- **PAT-003**: Component-based UI architecture
- **PAT-004**: Event-driven progress reporting
- **PAT-005**: Repository pattern for data access

## 2. Implementation Steps

### Implementation Phase 1: Project Setup & Core Backend

- GOAL-001: Set up the web UI project structure with FastAPI backend and static file serving

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-001 | Create `wakegen/web/` package with `__init__.py` and module structure | | |
| TASK-002 | Create `wakegen/web/app.py` with FastAPI application setup, CORS, static files | | |
| TASK-003 | Create `wakegen/web/config.py` for web server configuration (port, host, debug) | | |
| TASK-004 | Add web dependencies to `pyproject.toml`: `fastapi`, `uvicorn`, `jinja2`, `python-multipart`, `websockets` | | |
| TASK-005 | Create `wakegen/web/static/` directory for CSS, JS, and assets | | |
| TASK-006 | Create `wakegen/web/templates/` directory for Jinja2 HTML templates | | |
| TASK-007 | Create base template `templates/base.html` with Tailwind CSS, Alpine.js, and HTMX includes | | |
| TASK-008 | Add CLI command `wakegen serve` to start the web server | | |

### Implementation Phase 2: Provider Management API & UI

- GOAL-002: Create provider management endpoints and UI for viewing/configuring TTS providers

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-009 | Create `wakegen/web/routers/providers.py` with provider API endpoints | | |
| TASK-010 | Implement `GET /api/providers` - list all providers with availability status | | |
| TASK-011 | Implement `GET /api/providers/{provider_id}/voices` - list voices for a provider | | |
| TASK-012 | Implement `GET /api/providers/{provider_id}/status` - check provider availability | | |
| TASK-013 | Implement `POST /api/providers/{provider_id}/test` - test TTS generation with sample text | | |
| TASK-014 | Create `templates/pages/providers.html` - provider listing page with cards | | |
| TASK-015 | Create `templates/components/provider_card.html` - reusable provider card component | | |
| TASK-016 | Create `templates/components/voice_selector.html` - voice selection dropdown with search | | |
| TASK-017 | Add provider status indicators (available/unavailable/requires setup) with icons | | |
| TASK-018 | Add voice language filtering and search functionality | | |

### Implementation Phase 3: Configuration Editor

- GOAL-003: Create a visual YAML configuration editor with validation

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-019 | Create `wakegen/web/routers/config.py` with configuration API endpoints | | |
| TASK-020 | Implement `GET /api/config/template` - get template configuration | | |
| TASK-021 | Implement `POST /api/config/validate` - validate configuration JSON/YAML | | |
| TASK-022 | Implement `POST /api/config/save` - save configuration to file | | |
| TASK-023 | Implement `GET /api/config/load` - load existing configuration | | |
| TASK-024 | Implement `GET /api/config/presets` - list built-in config presets | | |
| TASK-025 | Create `templates/pages/config.html` - configuration editor page | | |
| TASK-026 | Create `templates/components/config_form.html` - form-based config editor with sections | | |
| TASK-027 | Implement project settings section (name, version, description) | | |
| TASK-028 | Implement generation settings section (wake words, count, output dir, sample rate) | | |
| TASK-029 | Implement provider selection section with drag-and-drop weight adjustment | | |
| TASK-030 | Implement augmentation settings section with profile selection | | |
| TASK-031 | Implement export settings section with format and split ratio controls | | |
| TASK-032 | Add real-time YAML preview panel | | |
| TASK-033 | Add validation error display with field highlighting | | |

### Implementation Phase 4: Generation Engine & Real-time Progress

- GOAL-004: Implement the core generation workflow with WebSocket-based progress reporting

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-034 | Create `wakegen/web/routers/generation.py` with generation API endpoints | | |
| TASK-035 | Create `wakegen/web/websocket.py` with WebSocket connection manager | | |
| TASK-036 | Implement `POST /api/generate/start` - start generation job | | |
| TASK-037 | Implement `GET /api/generate/status/{job_id}` - get job status | | |
| TASK-038 | Implement `POST /api/generate/cancel/{job_id}` - cancel running job | | |
| TASK-039 | Implement `WebSocket /ws/progress/{job_id}` - real-time progress updates | | |
| TASK-040 | Create `wakegen/web/services/generation_service.py` - background job management | | |
| TASK-041 | Integrate with existing `wakegen.generation.orchestrator` for actual generation | | |
| TASK-042 | Create `templates/pages/generate.html` - generation page | | |
| TASK-043 | Create `templates/components/generation_form.html` - quick generation form | | |
| TASK-044 | Create `templates/components/progress_bar.html` - real-time progress visualization | | |
| TASK-045 | Create `templates/components/job_history.html` - list of recent/running jobs | | |
| TASK-046 | Add audio preview for generated samples during/after generation | | |

### Implementation Phase 5: Augmentation Configuration UI

- GOAL-005: Create comprehensive augmentation configuration interface

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-047 | Create `wakegen/web/routers/augmentation.py` with augmentation API endpoints | | |
| TASK-048 | Implement `GET /api/augmentation/profiles` - list environment profiles | | |
| TASK-049 | Implement `GET /api/augmentation/profiles/{profile_id}` - get profile details | | |
| TASK-050 | Implement `GET /api/augmentation/devices` - list device presets | | |
| TASK-051 | Implement `GET /api/augmentation/devices/{device_id}` - get device preset details | | |
| TASK-052 | Implement `POST /api/augmentation/preview` - preview augmentation on sample audio | | |
| TASK-053 | Create `templates/pages/augmentation.html` - augmentation configuration page | | |
| TASK-054 | Create `templates/components/profile_card.html` - environment profile card | | |
| TASK-055 | Create `templates/components/device_preset_card.html` - device preset card | | |
| TASK-056 | Create noise profile visualizer with SNR slider | | |
| TASK-057 | Create room simulation visualizer (RT60, room size controls) | | |
| TASK-058 | Create microphone profile selector with frequency response display | | |
| TASK-059 | Add before/after audio comparison for augmentation preview | | |
| TASK-060 | Create waveform and spectrogram visualization component | | |

### Implementation Phase 6: Quality Assurance Dashboard

- GOAL-006: Create quality monitoring and validation interface

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-061 | Create `wakegen/web/routers/quality.py` with quality API endpoints | | |
| TASK-062 | Implement `POST /api/quality/validate` - validate audio samples | | |
| TASK-063 | Implement `GET /api/quality/stats/{dataset_dir}` - get dataset statistics | | |
| TASK-064 | Implement `POST /api/quality/asr-check` - run ASR verification | | |
| TASK-065 | Implement `GET /api/quality/report/{dataset_dir}` - generate quality report | | |
| TASK-066 | Create `templates/pages/quality.html` - quality dashboard page | | |
| TASK-067 | Create `templates/components/quality_metrics.html` - metrics display cards | | |
| TASK-068 | Create `templates/components/validation_results.html` - validation results table | | |
| TASK-069 | Create `templates/components/statistics_charts.html` - dataset statistics charts | | |
| TASK-070 | Add sample browser with quality filtering (good/bad/needs review) | | |
| TASK-071 | Add batch validation with progress indicator | | |

### Implementation Phase 7: Export & Dataset Management

- GOAL-007: Create export interface for all supported formats

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-072 | Create `wakegen/web/routers/export.py` with export API endpoints | | |
| TASK-073 | Implement `GET /api/export/formats` - list available export formats | | |
| TASK-074 | Implement `POST /api/export/start` - start export job | | |
| TASK-075 | Implement `GET /api/export/download/{job_id}` - download exported dataset | | |
| TASK-076 | Create `templates/pages/export.html` - export configuration page | | |
| TASK-077 | Create `templates/components/export_format_card.html` - format selection cards | | |
| TASK-078 | Create `templates/components/split_ratio_slider.html` - train/val/test split controls | | |
| TASK-079 | Add export preview showing expected directory structure | | |
| TASK-080 | Add export progress with file count and size estimates | | |

### Implementation Phase 8: System Status & Utilities

- GOAL-008: Create system monitoring and utility interfaces

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-081 | Create `wakegen/web/routers/system.py` with system API endpoints | | |
| TASK-082 | Implement `GET /api/system/gpu` - get GPU status | | |
| TASK-083 | Implement `GET /api/system/cache/stats` - get cache statistics | | |
| TASK-084 | Implement `POST /api/system/cache/clear` - clear cache | | |
| TASK-085 | Implement `GET /api/system/cache/entries` - list cache entries | | |
| TASK-086 | Implement `GET /api/system/health` - health check endpoint | | |
| TASK-087 | Create `templates/pages/system.html` - system status page | | |
| TASK-088 | Create `templates/components/gpu_status.html` - GPU status card | | |
| TASK-089 | Create `templates/components/cache_stats.html` - cache statistics card | | |
| TASK-090 | Create `templates/components/cache_entries.html` - cache entries table | | |
| TASK-091 | Add disk usage visualization | | |

### Implementation Phase 9: Dashboard & Navigation

- GOAL-009: Create main dashboard and navigation structure

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-092 | Create `templates/pages/dashboard.html` - main dashboard page | | |
| TASK-093 | Create `templates/components/sidebar.html` - navigation sidebar | | |
| TASK-094 | Create `templates/components/header.html` - header with theme toggle | | |
| TASK-095 | Create dashboard summary cards (providers, recent jobs, cache, GPU) | | |
| TASK-096 | Create quick actions panel (new generation, new config, export) | | |
| TASK-097 | Create recent activity feed showing job completions | | |
| TASK-098 | Implement dark/light theme toggle with persistence | | |
| TASK-099 | Add keyboard shortcuts for common actions | | |
| TASK-100 | Create 404 and error pages | | |

### Implementation Phase 10: Audio Player & Visualization

- GOAL-010: Create audio playback and visualization components

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-101 | Create `wakegen/web/routers/audio.py` with audio API endpoints | | |
| TASK-102 | Implement `GET /api/audio/{path}` - serve audio files | | |
| TASK-103 | Implement `GET /api/audio/waveform/{path}` - get waveform data | | |
| TASK-104 | Create `templates/components/audio_player.html` - custom audio player | | |
| TASK-105 | Create `templates/components/waveform.html` - waveform visualization using Canvas/SVG | | |
| TASK-106 | Create `templates/components/spectrogram.html` - spectrogram visualization | | |
| TASK-107 | Add playback controls (play, pause, seek, volume) | | |
| TASK-108 | Add batch playback for comparing samples | | |

### Implementation Phase 11: Testing & Documentation

- GOAL-011: Add tests and documentation for the Web UI

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-109 | Create `tests/web/` directory for web UI tests | | |
| TASK-110 | Write unit tests for API endpoints | | |
| TASK-111 | Write integration tests for WebSocket progress | | |
| TASK-112 | Write E2E tests using Playwright or similar | | |
| TASK-113 | Update `docs/` with Web UI usage documentation | | |
| TASK-114 | Add API documentation using OpenAPI/Swagger | | |
| TASK-115 | Create screenshots and usage GIFs for README | | |

### Implementation Phase 12: Polish & Performance

- GOAL-012: Final polish, optimizations, and UX improvements

| Task     | Description           | Completed | Date |
| -------- | --------------------- | --------- | ---- |
| TASK-116 | Add loading skeletons for async content | | |
| TASK-117 | Implement error boundary and toast notifications | | |
| TASK-118 | Add confirmation dialogs for destructive actions | | |
| TASK-119 | Optimize static asset loading (minification, bundling) | | |
| TASK-120 | Add PWA manifest for installability | | |
| TASK-121 | Performance testing and optimization | | |
| TASK-122 | Cross-browser testing (Chrome, Firefox, Safari) | | |
| TASK-123 | Accessibility audit and improvements (ARIA labels, keyboard nav) | | |

## 3. Alternatives

- **ALT-001**: **NiceGUI** - Pure Python UI framework. Pros: No frontend build, reactive, Quasar-based. Cons: Less flexibility, larger dependency.
- **ALT-002**: **Streamlit** - Data science focused. Pros: Very quick to build. Cons: Limited customization, not ideal for complex forms.
- **ALT-003**: **Gradio** - ML-focused UI. Pros: Great for ML demos. Cons: Limited layout control, not ideal for complex workflows.
- **ALT-004**: **React/Vue SPA** - Full SPA frontend. Pros: Maximum flexibility, rich ecosystem. Cons: Separate build process, more complexity.
- **ALT-005**: **Flask instead of FastAPI** - Traditional WSGI. Pros: More mature. Cons: No native async, slower for our use case.

**Chosen Approach**: FastAPI + Jinja2 + TailwindCSS + Alpine.js + HTMX

**Rationale**: This stack provides:
- Native async support (critical for WebSockets and long-running generation)
- Server-side rendering with progressive enhancement
- Minimal JavaScript while maintaining rich interactivity
- Easy integration with existing Python codebase
- No separate build process for frontend
- Lightweight and fast

## 4. Dependencies

- **DEP-001**: `fastapi>=0.109.0` - Modern async web framework
- **DEP-002**: `uvicorn[standard]>=0.27.0` - ASGI server with WebSocket support
- **DEP-003**: `jinja2>=3.1.0` - Template engine (already installed)
- **DEP-004**: `python-multipart>=0.0.6` - File upload support
- **DEP-005**: `websockets>=12.0` - WebSocket protocol support
- **DEP-006**: `aiofiles>=23.0.0` - Async file operations
- **DEP-007**: TailwindCSS (via CDN) - Utility-first CSS framework
- **DEP-008**: Alpine.js (via CDN) - Minimal reactive framework
- **DEP-009**: HTMX (via CDN) - HTML-over-the-wire for AJAX

## 5. Files

### New Files to Create

- **FILE-001**: `wakegen/web/__init__.py` - Web module initialization
- **FILE-002**: `wakegen/web/app.py` - FastAPI application factory and configuration
- **FILE-003**: `wakegen/web/config.py` - Web server configuration settings
- **FILE-004**: `wakegen/web/websocket.py` - WebSocket connection manager
- **FILE-005**: `wakegen/web/routers/__init__.py` - Router package
- **FILE-006**: `wakegen/web/routers/providers.py` - Provider API endpoints
- **FILE-007**: `wakegen/web/routers/config.py` - Configuration API endpoints
- **FILE-008**: `wakegen/web/routers/generation.py` - Generation API endpoints
- **FILE-009**: `wakegen/web/routers/augmentation.py` - Augmentation API endpoints
- **FILE-010**: `wakegen/web/routers/quality.py` - Quality API endpoints
- **FILE-011**: `wakegen/web/routers/export.py` - Export API endpoints
- **FILE-012**: `wakegen/web/routers/system.py` - System status API endpoints
- **FILE-013**: `wakegen/web/routers/audio.py` - Audio serving endpoints
- **FILE-014**: `wakegen/web/services/__init__.py` - Services package
- **FILE-015**: `wakegen/web/services/generation_service.py` - Background job service
- **FILE-016**: `wakegen/web/services/audio_service.py` - Audio processing service
- **FILE-017**: `wakegen/web/templates/base.html` - Base HTML template
- **FILE-018**: `wakegen/web/templates/pages/dashboard.html` - Dashboard page
- **FILE-019**: `wakegen/web/templates/pages/providers.html` - Providers page
- **FILE-020**: `wakegen/web/templates/pages/config.html` - Configuration page
- **FILE-021**: `wakegen/web/templates/pages/generate.html` - Generation page
- **FILE-022**: `wakegen/web/templates/pages/augmentation.html` - Augmentation page
- **FILE-023**: `wakegen/web/templates/pages/quality.html` - Quality page
- **FILE-024**: `wakegen/web/templates/pages/export.html` - Export page
- **FILE-025**: `wakegen/web/templates/pages/system.html` - System page
- **FILE-026**: `wakegen/web/templates/components/*.html` - Various reusable components
- **FILE-027**: `wakegen/web/static/css/custom.css` - Custom CSS overrides
- **FILE-028**: `wakegen/web/static/js/app.js` - Application JavaScript
- **FILE-029**: `wakegen/web/static/js/websocket.js` - WebSocket client
- **FILE-030**: `wakegen/web/static/js/audio-player.js` - Audio player logic
- **FILE-031**: `wakegen/web/static/js/waveform.js` - Waveform visualization

### Files to Modify

- **FILE-032**: `pyproject.toml` - Add web dependencies
- **FILE-033**: `wakegen/main.py` - Add `serve` command import
- **FILE-034**: `wakegen/ui/cli/commands.py` - Add `serve` CLI command

## 6. Testing

- **TEST-001**: Unit tests for all API endpoint handlers
- **TEST-002**: Integration tests for WebSocket progress reporting
- **TEST-003**: Integration tests for generation workflow via web API
- **TEST-004**: Unit tests for configuration validation endpoints
- **TEST-005**: Integration tests for audio serving and playback
- **TEST-006**: E2E tests for complete generation workflow
- **TEST-007**: E2E tests for configuration creation and validation
- **TEST-008**: Performance tests for concurrent WebSocket connections
- **TEST-009**: Load tests for audio file serving
- **TEST-010**: Accessibility tests using axe-core or similar

## 7. Risks & Assumptions

### Risks

- **RISK-001**: WebSocket connections may be unreliable on slow networks - Mitigate with fallback polling
- **RISK-002**: Large audio file transfers may timeout - Mitigate with chunked streaming
- **RISK-003**: Long-running generation jobs may be interrupted by server restart - Mitigate with job persistence
- **RISK-004**: Browser compatibility issues with audio APIs - Mitigate with Web Audio API polyfills
- **RISK-005**: Memory usage may grow with many concurrent WebSocket connections - Monitor and implement connection limits

### Assumptions

- **ASSUMPTION-001**: Users have modern browsers (Chrome, Firefox, Safari, Edge latest 2 versions)
- **ASSUMPTION-002**: Users have sufficient disk space for generated audio files
- **ASSUMPTION-003**: Web server runs on localhost or trusted network (no external access by default)
- **ASSUMPTION-004**: Python 3.10+ is available (consistent with project requirements)
- **ASSUMPTION-005**: uvicorn can be installed and run on the target platform

## 8. Related Specifications / Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HTMX Documentation](https://htmx.org/docs/)
- [Alpine.js Documentation](https://alpinejs.dev/)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [WakeGen README](../README.md)
- [WakeGen Configuration Reference](../docs/configuration.md)
- [WakeGen Provider Documentation](../docs/providers.md)

---

## UI Wireframes Reference

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  🎤 WakeGen                                    [🌙] [Settings]  │
├────────────┬────────────────────────────────────────────────────┤
│            │  ┌─────────────────────────────────────────────┐   │
│ Dashboard  │  │  Welcome to WakeGen                         │   │
│            │  │  Generate high-quality wake word datasets   │   │
│ Providers  │  └─────────────────────────────────────────────┘   │
│            │                                                     │
│ Generate   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│            │  │Providers │ │ Samples  │ │  Cache   │ │  GPU   │ │
│ Augment    │  │  8/11 ✓  │ │   1,234  │ │  45 MB   │ │  RTX   │ │
│            │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│ Quality    │                                                     │
│            │  ┌─────────────────────────────────────────────┐   │
│ Export     │  │  Quick Actions                               │   │
│            │  │  [+ New Generation] [📁 Load Config] [⬇ Ex] │   │
│ Config     │  └─────────────────────────────────────────────┘   │
│            │                                                     │
│ System     │  ┌─────────────────────────────────────────────┐   │
│            │  │  Recent Jobs                                 │   │
│            │  │  ──────────────────────────────────────────  │   │
│            │  │  ✓ hey_katya - 500 samples - 2 min ago      │   │
│            │  │  ✓ jarvis - 200 samples - 1 hour ago        │   │
│            │  └─────────────────────────────────────────────┘   │
└────────────┴────────────────────────────────────────────────────┘
```

### Provider Card Design
```
┌────────────────────────────────────────────┐
│  ┌────┐                                    │
│  │ 🎙 │  Edge TTS                    ✓ ON │
│  └────┘  Microsoft Edge Text-to-Speech     │
│                                            │
│  Type: Cloud │ GPU: No │ Quality: ⭐⭐⭐⭐ │
│                                            │
│  Voices: 400+ │ Languages: 50+             │
│                                            │
│  [Configure] [Test Voice] [List Voices]    │
└────────────────────────────────────────────┘
```

### Generation Progress
```
┌────────────────────────────────────────────────────────────────┐
│  Generating: "hey assistant"                                   │
│                                                                │
│  ████████████████████████░░░░░░░░░░  65% (650/1000)           │
│                                                                │
│  Provider: edge_tts │ Voice: en-US-JennyNeural                │
│  Current: hey_assistant_0651.wav                               │
│                                                                │
│  ⏱ Elapsed: 2:34 │ Remaining: ~1:20                           │
│                                                                │
│  [Pause] [Cancel]                                              │
└────────────────────────────────────────────────────────────────┘
```
