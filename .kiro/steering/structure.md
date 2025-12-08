# WakeGen - Project Structure

```
wakegen/
├── main.py              # Entry point, imports CLI
├── __init__.py          # Package init, version, plugin loading
│
├── core/                # Foundation types and contracts
│   ├── types.py         # Enums (ProviderType, AudioFormat, Gender, etc.)
│   ├── protocols.py     # TTSProvider protocol (interface definition)
│   └── exceptions.py    # Custom exceptions (ConfigError, ProviderError, etc.)
│
├── models/              # Pydantic data models
│   ├── config.py        # ProviderConfig, GenerationConfig (BaseSettings)
│   ├── audio.py         # Voice, AudioSample models
│   └── generation.py    # GenerationParameters, GenerationResult
│
├── providers/           # TTS provider implementations
│   ├── base.py          # BaseProvider ABC
│   ├── registry.py      # Provider registration and factory
│   ├── free/            # Free providers (edge_tts.py)
│   ├── commercial/      # Commercial providers (minimax.py)
│   └── opensource/      # Open source providers (bark, piper, kokoro, etc.)
│
├── generation/          # Generation orchestration
│   ├── orchestrator.py  # Main coordinator
│   ├── batch_processor.py  # Concurrent task processing
│   ├── checkpoint.py    # Resume capability (SQLite-based)
│   ├── progress.py      # Rich progress tracking
│   ├── rate_limiter.py  # API rate limiting
│   └── variation_engine.py  # Parameter variation generation
│
├── augmentation/        # Audio augmentation pipeline
│   ├── pipeline.py      # Main augmentation orchestrator
│   ├── profiles.py      # Environment profiles
│   ├── effects/         # Time/frequency domain effects
│   ├── noise/           # Noise injection and mixing
│   ├── room/            # Room impulse response simulation
│   └── microphone/      # Microphone simulation
│
├── quality/             # Quality assurance
│   ├── validator.py     # Dataset validation
│   ├── scorer.py        # Quality scoring
│   ├── asr_check.py     # ASR verification
│   ├── deduplication.py # Duplicate detection
│   └── statistics.py    # Dataset statistics
│
├── export/              # Export format handlers
│   ├── formats.py       # Format definitions
│   ├── manifest.py      # Manifest generation
│   ├── splitter.py      # Train/val/test splitting
│   └── openwakeword.py  # OpenWakeWord format
│
├── training/            # Training script generation
│   ├── script_generator.py  # Generate training scripts
│   ├── model_tester.py  # Model testing utilities
│   └── ab_comparison.py # A/B comparison tools
│
├── config/              # Configuration management
│   ├── settings.py      # Settings loaders
│   ├── yaml_loader.py   # YAML config parsing
│   └── presets/         # Pre-built YAML configs
│
├── ui/                  # User interfaces
│   └── cli/             # Click-based CLI
│       ├── commands.py  # CLI command definitions
│       └── wizard.py    # Interactive wizard
│
├── web/                 # Optional web UI (FastAPI)
│   ├── app.py           # FastAPI application
│   ├── routers/         # API route handlers
│   ├── templates/       # Jinja2 HTML templates
│   └── static/          # CSS, JS assets
│
├── plugins/             # Plugin system
│   ├── base.py          # Plugin base class
│   └── discovery.py     # Plugin auto-discovery
│
└── utils/               # Shared utilities
    ├── audio.py         # Audio processing helpers
    ├── caching.py       # Generation cache
    ├── logging.py       # Logging setup
    ├── async_helpers.py # Async utilities
    └── gpu.py           # GPU detection
```

## Key Patterns

### Provider Pattern
All TTS providers implement `TTSProvider` protocol and extend `BaseProvider`:
- Register via `@register_provider(ProviderType.X, XProvider)`
- Factory access via `get_provider(provider_type, config)`

### Configuration
- `BaseSettings` from pydantic-settings for env var loading
- YAML configs parsed via `load_config()` in yaml_loader.py
- Validation aliases map env vars (e.g., `MINIMAX_API_KEY`)

### Async Throughout
- All provider methods are async (`async def generate()`, `async def list_voices()`)
- CLI uses `asyncio.run()` to bridge sync/async
- Batch processing uses asyncio for concurrency
