import click
import asyncio
import os
from pathlib import Path
from typing import Optional, Any
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.table import Table
from wakegen.config.settings import get_generation_config, get_provider_config
from wakegen.config.yaml_loader import load_config, get_template_config, WakegenConfig
from wakegen.providers.registry import (
    get_provider,
    list_available_providers,
    discover_available_providers,
    ProviderInfo,
)
from wakegen.core.types import ProviderType
from wakegen.core.protocols import TTSProvider
from wakegen.core.exceptions import ConfigError
from wakegen.utils.logging import setup_logging
from wakegen.utils.audio import resample_audio
from wakegen.ui.cli.wizard import run_wizard
# Import providers to ensure they are registered
import wakegen.providers

# We use 'click' to create the command line interface.
# It handles parsing arguments (like --text "hello") and displaying help messages.
# We use 'rich' for beautiful output.

console = Console()

@click.group()
def cli():
    """
    Wake Word Dataset Generator CLI.
    
    This tool helps you generate, augment, and prepare datasets for training
    custom wake word models (like "Hey Computer").
    """
    setup_logging()

@cli.command(name="list-providers")
@click.option("--available-only", "-a", is_flag=True, help="Show only providers that are ready to use")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information including install hints")
def list_providers(available_only: bool, verbose: bool):
    """
    Lists all supported TTS providers and their availability status.
    
    This command checks which providers are actually usable based on:
    - Required Python packages being installed
    - API keys being configured (for commercial providers)
    
    Use this to see which providers you can use with the --provider flag,
    and what you need to install to enable more providers.
    
    Examples:
        wakegen list-providers              # Show all providers
        wakegen list-providers --available-only  # Show only usable ones
        wakegen list-providers --verbose    # Show install instructions
    """
    # Discover all providers and their availability status
    providers = discover_available_providers()
    
    # Filter if requested
    if available_only:
        providers = [p for p in providers if p.is_available]
    
    if not providers:
        console.print("[yellow]No providers available.[/yellow]")
        console.print("Use [cyan]wakegen list-providers --verbose[/cyan] for installation instructions.")
        return
    
    # Create a nice table
    table = Table(title="TTS Providers", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Description")
    if verbose:
        table.add_column("Requirements")
    
    for p in providers:
        # Status indicator
        if p.is_available:
            status = "[green]✓ Available[/green]"
        else:
            status = "[red]✗ Not Available[/red]"
        
        # Build the row
        if verbose:
            if p.is_available:
                requirements = "[dim]All satisfied[/dim]"
            else:
                requirements = "\n".join([
                    f"[yellow]• {dep}[/yellow]" for dep in p.missing_dependencies
                ])
                if p.install_hint:
                    requirements += f"\n[dim]{p.install_hint}[/dim]"
            table.add_row(p.name, status, p.description, requirements)
        else:
            table.add_row(p.name, status, p.description)
    
    console.print(table)
    
    # Summary
    available_count = sum(1 for p in providers if p.is_available)
    total_count = len(providers)
    console.print(f"\n[dim]{available_count}/{total_count} providers available[/dim]")
    
    if not verbose and available_count < total_count:
        console.print("[dim]Use --verbose to see installation instructions for missing providers.[/dim]")

@cli.command(name="list-voices")
@click.option("--provider", default="all", help="Provider to list voices for (default: all)")
def list_voices(provider: str):
    """
    Lists available voices for a specific provider or all providers.
    
    Example: wakegen list-voices --provider piper
    """
    asyncio.run(run_list_voices(provider))

async def run_list_voices(provider_name: str):
    """
    Async implementation of list-voices command.
    """
    provider_config = get_provider_config()
    
    # Determine which providers to query
    if provider_name.lower() == "all":
        providers_to_check = list_available_providers()
    else:
        try:
            # Convert string input to ProviderType enum
            providers_to_check = [ProviderType(provider_name.lower())]
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider_name}'")
            console.print("Use 'wakegen list-providers' to see available options.")
            return

    for p_type in providers_to_check:
        try:
            console.print(f"\n[bold cyan]Fetching voices for {p_type.value}...[/bold cyan]")
            # Instantiate the provider
            provider = get_provider(p_type, provider_config)
            # Fetch voices asynchronously
            voices = await provider.list_voices()
            
            if not voices:
                console.print(f"  [yellow]No voices found for {p_type.value}[/yellow]")
                continue

            # Display voices in a table-like format
            console.print(f"  Found {len(voices)} voices:")
            for voice in voices:
                console.print(f"  - [green]{voice.name}[/green] (ID: {voice.id}) - {voice.language}")
                
        except Exception as e:
            console.print(f"  [red]Failed to fetch voices for {p_type.value}: {str(e)}[/red]")

@cli.command()
@click.option("--text", help="The wake word text to generate (e.g., 'Hey Katya')")
@click.option("--count", default=1, help="Number of samples to generate")
@click.option("--preset", help="Name of a configuration preset to use (e.g., 'quick_test')")
@click.option("--output-dir", help="Directory to save the output")
@click.option("--provider", default="edge_tts", help="TTS provider to use (default: edge_tts)")
@click.option("--voice", help="Specific voice ID to use")
@click.option("--interactive", is_flag=True, help="Run in interactive wizard mode")
@click.option(
    "--config",
    "-c",
    "config_file",  # Use different name to avoid shadowing the config command group
    type=click.Path(exists=True),
    help="Path to wakegen.yaml configuration file"
)
@click.option(
    "--cache/--no-cache",
    "use_cache",
    default=True,
    help="Enable/disable caching of generated audio (default: enabled)"
)
def generate(
    text: Optional[str],
    count: int,
    preset: Optional[str],
    output_dir: Optional[str],
    provider: str,
    voice: Optional[str],
    interactive: bool,
    config_file: Optional[str],
    use_cache: bool
):
    """
    Generates audio samples for a given wake word.
    
    You can provide arguments directly, use --interactive for a guided wizard,
    or use --config to load settings from a YAML file.
    
    Configuration file takes precedence, but CLI flags can override specific values.
    
    Caching is enabled by default. When enabled, identical generation requests
    (same text + voice + provider) will reuse previously generated audio files,
    saving time and API costs. Use --no-cache to always regenerate fresh samples.
    
    Examples:
        wakegen generate --text "hey assistant" --count 10
        wakegen generate --config wakegen.yaml
        wakegen generate --config wakegen.yaml --count 50  # Override count
        wakegen generate --interactive
        wakegen generate --text "hey jarvis" --no-cache  # Force regeneration
    """
    # If config file is provided, use config-based generation
    if config_file:
        asyncio.run(run_generation_from_config(config_file, text, count, output_dir, provider, voice, use_cache))
    # If interactive flag is set OR no text is provided (and no config), run the wizard
    elif interactive or not text:
        asyncio.run(run_interactive_generation())
    else:
        # Run the async function in the event loop
        asyncio.run(run_generation(text, count, preset, output_dir, provider, voice, use_cache))

async def run_interactive_generation():
    """
    Runs the generation process using the interactive wizard.
    """
    config = await run_wizard()
    
    if not config:
        return

    # Map wizard config to generation arguments
    await run_generation(
        text=config["wake_word"],
        count=config["count"],
        preset=None, # Wizard builds custom config
        output_dir=config["output_dir"],
        provider_name="edge_tts", # Default for wizard for now
        voice_id=None
    )


async def run_generation_from_config(
    config_path: str,
    text_override: Optional[str] = None,
    count_override: Optional[int] = None,
    output_dir_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    voice_override: Optional[str] = None,
    use_cache: bool = True
):
    """
    Run generation using a YAML configuration file.
    
    This is the main config-driven generation flow. It reads the config file,
    applies any CLI overrides, and generates samples for each wake word using
    each configured provider based on their weights.
    
    Think of it like a recipe executor:
    1. Read the recipe (config file)
    2. Apply any last-minute changes (CLI overrides)
    3. For each ingredient (wake word), use the right tools (providers) in the right proportions (weights)
    
    Args:
        config_path: Path to wakegen.yaml configuration file.
        text_override: Override wake words from config (generates only this word).
        count_override: Override sample count from config.
        output_dir_override: Override output directory from config.
        provider_override: Override to use only this provider (ignores config providers).
        voice_override: Override to use only this voice ID.
        use_cache: Whether to use caching for generated audio.
    """
    # Initialize cache if enabled
    from wakegen.utils.caching import GenerationCache
    cache = GenerationCache(enabled=use_cache)
    
    if use_cache:
        console.print("[dim]Caching enabled (use --no-cache to disable)[/dim]")
    
    try:
        # 1. Load and validate configuration
        console.print(f"[dim]Loading configuration from:[/dim] {config_path}")
        config = load_config(config_path)
        
        # 2. Apply CLI overrides
        # Wake words: use override or config value
        wake_words = [text_override] if text_override else config.generation.wake_words
        
        # Count: use override, fall back to config, default 1
        count = count_override if count_override and count_override > 1 else config.generation.count
        
        # Output directory: use override or config value
        output_dir = output_dir_override if output_dir_override else config.generation.output_dir
        
        # 3. Display generation plan
        console.print(Panel(
            f"[bold]Project:[/bold] {config.project.name} v{config.project.version}\n"
            f"[bold]Wake Words:[/bold] {', '.join(wake_words)}\n"
            f"[bold]Count per word:[/bold] {count}\n"
            f"[bold]Output:[/bold] {output_dir}\n"
            f"[bold]Audio:[/bold] {config.generation.audio_format} @ {config.generation.sample_rate}Hz",
            title="Generation Plan",
            border_style="cyan"
        ))
        
        # 4. Determine which providers to use
        if provider_override:
            # CLI override: use only the specified provider
            try:
                provider_type = ProviderType(provider_override.lower())
                providers_to_use = [(provider_type, 1.0, voice_override)]
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider_override}'")
                console.print("Use 'wakegen list-providers' to see available options.")
                return
        else:
            # Use providers from config with their weights
            providers_to_use = []
            for p_config in config.providers:
                try:
                    provider_type = ProviderType(p_config.type.lower())
                    # If voice override provided, use it; otherwise use first voice from config or None
                    voice = voice_override or (p_config.voices[0] if p_config.voices else None)
                    providers_to_use.append((provider_type, p_config.weight, voice))
                except ValueError:
                    console.print(f"[yellow]Warning:[/yellow] Skipping unknown provider '{p_config.type}'")
        
        if not providers_to_use:
            console.print("[bold red]Error:[/bold red] No valid providers configured.")
            return
        
        # 5. Show provider breakdown
        console.print("\n[bold]Provider Distribution:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Provider")
        table.add_column("Weight")
        table.add_column("Samples/Word")
        table.add_column("Voice")
        
        for p_type, weight, voice_id in providers_to_use:
            samples = int(count * weight)
            voice_str = voice_id if voice_id else "[auto]"
            table.add_row(p_type.value, f"{weight:.0%}", str(samples), voice_str)
        
        console.print(table)
        console.print("")
        
        # 6. Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 7. Generate samples for each wake word
        provider_config = get_provider_config()
        total_generated = 0
        total_failed = 0
        
        for wake_word in wake_words:
            console.print(f"\n[bold cyan]Generating samples for:[/bold cyan] '{wake_word}'")
            
            # Create subdirectory for this wake word
            word_dir = os.path.join(output_dir, wake_word.replace(" ", "_").lower())
            os.makedirs(word_dir, exist_ok=True)
            
            sample_index = 0
            
            for p_type, weight, preferred_voice in providers_to_use:
                # Calculate how many samples this provider should generate
                provider_count = int(count * weight)
                if provider_count == 0:
                    continue
                
                try:
                    # Get provider instance
                    provider = get_provider(p_type, provider_config)
                    
                    # Get voice to use
                    selected_voice = None
                    if preferred_voice:
                        # User specified a voice
                        from wakegen.core.protocols import Voice
                        from wakegen.core.types import Gender
                        selected_voice = Voice(
                            id=preferred_voice,
                            name=preferred_voice,
                            language="unknown",
                            gender=Gender.NEUTRAL
                        )
                    else:
                        # Auto-select a voice
                        voices = await provider.list_voices()
                        if voices:
                            # Prefer English neural voice, fallback to first available
                            selected_voice = next(
                                (v for v in voices if v.language.startswith("en-")),
                                voices[0]
                            )
                    
                    if not selected_voice:
                        console.print(f"  [yellow]No voice available for {p_type.value}, skipping[/yellow]")
                        continue
                    
                    console.print(f"  Using [cyan]{p_type.value}[/cyan] with voice '{selected_voice.name}'")
                    
                    # Generate samples with progress
                    cache_hits = 0
                    for i in track(range(provider_count), description=f"  {p_type.value}"):
                        sample_index += 1
                        filename = f"{wake_word.replace(' ', '_').lower()}_{sample_index:04d}_{p_type.value}.{config.generation.audio_format}"
                        file_path = os.path.join(word_dir, filename)
                        
                        try:
                            # Check cache first
                            cached_path = cache.get(wake_word, selected_voice.id, p_type.value)
                            if cached_path:
                                # Copy from cache to target location
                                import shutil
                                shutil.copy2(cached_path, file_path)
                                cache_hits += 1
                            else:
                                # Generate new audio
                                await provider.generate(wake_word, selected_voice.id, file_path)
                                # Add to cache
                                cache.put(wake_word, selected_voice.id, p_type.value, file_path, copy=True)
                            
                            # Resample if needed
                            if config.generation.sample_rate:
                                resample_audio(file_path, config.generation.sample_rate)
                            
                            total_generated += 1
                            
                        except Exception as e:
                            console.print(f"    [red]Failed sample {sample_index}: {e}[/red]")
                            total_failed += 1
                    
                    # Report cache hits for this provider
                    if cache_hits > 0:
                        console.print(f"    [dim]({cache_hits} from cache)[/dim]")
                            
                except Exception as e:
                    console.print(f"  [red]Provider {p_type.value} failed: {e}[/red]")
                    total_failed += provider_count
        
        # Save cache metadata
        cache.save_metadata()
        cache_stats = cache.get_stats()
        
        # 8. Show summary
        cache_info = ""
        if use_cache:
            cache_info = f"\n[bold]Cache:[/bold] {cache_stats.hits} hits, {cache_stats.misses} new"
        
        console.print(Panel(
            f"[bold green]✓ Generation complete![/bold green]\n\n"
            f"[bold]Total generated:[/bold] {total_generated}\n"
            f"[bold]Failed:[/bold] {total_failed}\n"
            f"[bold]Output directory:[/bold] {output_dir}{cache_info}",
            title="Summary",
            border_style="green"
        ))
        
    except ConfigError as e:
        console.print(Panel(
            f"[bold red]Configuration Error[/bold red]\n\n{e}",
            border_style="red"
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())

async def run_generation(text: str, count: int, preset: Optional[str], output_dir: Optional[str], provider_name: str = "edge_tts", voice_id: Optional[str] = None, use_cache: bool = True):
    """
    The main logic for the generation command.
    
    Args:
        text: The wake word text to generate.
        count: Number of samples to generate.
        preset: Configuration preset name.
        output_dir: Output directory path.
        provider_name: TTS provider to use.
        voice_id: Specific voice ID to use.
        use_cache: Whether to use caching.
    """
    # Initialize cache
    from wakegen.utils.caching import GenerationCache
    cache = GenerationCache(enabled=use_cache)
    
    if use_cache:
        console.print("[dim]Caching enabled (use --no-cache to disable)[/dim]")
    
    try:
        # 1. Load Configuration
        # If a preset is given, load it. Otherwise, use defaults.
        gen_config = get_generation_config(preset) if preset else get_generation_config("quick_test")
        provider_config = get_provider_config()

        # Override output directory if provided via CLI
        if output_dir:
            gen_config.output_dir = output_dir

        console.print(f"[bold green]Starting generation for:[/bold green] '{text}'")
        console.print(f"Output directory: {gen_config.output_dir}")

        # 2. Get the Provider
        try:
            provider_type = ProviderType(provider_name.lower())
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider_name}'")
            console.print("Use 'wakegen list-providers' to see available options.")
            return

        console.print(f"Using provider: [cyan]{provider_type.value}[/cyan]")
        provider = get_provider(provider_type, provider_config)

        # 3. Get a voice to use
        selected_voice = None
        
        if voice_id:
            # If user specified a voice, try to find it to validate
            console.print(f"Verifying voice '{voice_id}'...")
            voices = await provider.list_voices()
            selected_voice = next((v for v in voices if v.id == voice_id or v.name == voice_id), None)
            
            if not selected_voice:
                console.print(f"[yellow]Warning:[/yellow] Voice '{voice_id}' not found in list, attempting to use anyway...")
                # Create a dummy voice object if we can't find it but user insisted
                # This supports providers where list_voices might be incomplete or slow
                from wakegen.core.protocols import Voice
                from wakegen.core.types import Gender
                selected_voice = Voice(id=voice_id, name=voice_id, language="unknown", gender=Gender.NEUTRAL)
        else:
            # Auto-select a voice
            console.print("Fetching available voices...")
            voices = await provider.list_voices()
            
            # Simple filter for an English voice
            # We prioritize English Neural voices for Edge TTS, or just the first available for others
            selected_voice = next((v for v in voices if v.language.startswith("en-") and "Neural" in v.id), None)
            
            # Fallback: just take the first one
            if not selected_voice and voices:
                selected_voice = voices[0]
        
        if not selected_voice:
            console.print("[bold red]No suitable voice found![/bold red]")
            return

        console.print(f"Using voice: [cyan]{selected_voice.name}[/cyan] ({selected_voice.id})")

        # 4. Generate Samples
        os.makedirs(gen_config.output_dir, exist_ok=True)
        
        cache_hits = 0
        for i in track(range(count), description="Generating samples..."):
            filename = f"{text.replace(' ', '_').lower()}_{i+1}.{gen_config.audio_format}"
            file_path = os.path.join(gen_config.output_dir, filename)
            
            # Check cache first
            cached_path = cache.get(text, selected_voice.id, provider_type.value)
            if cached_path:
                # Copy from cache to target location
                import shutil
                shutil.copy2(cached_path, file_path)
                cache_hits += 1
            else:
                # Generate the audio
                await provider.generate(text, selected_voice.id, file_path)
                # Add to cache
                cache.put(text, selected_voice.id, provider_type.value, file_path, copy=True)
            
            # Resample if needed (Edge TTS usually outputs 24kHz, we might want 16kHz)
            if gen_config.sample_rate:
                resample_audio(file_path, gen_config.sample_rate)
        
        # Save cache metadata
        cache.save_metadata()
        
        # Report results
        if cache_hits > 0:
            console.print(f"[bold green]Successfully generated {count} samples![/bold green] [dim]({cache_hits} from cache)[/dim]")
        else:
            console.print(f"[bold green]Successfully generated {count} samples![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        # In a real app, we might want to print the full traceback here
        # console.print_exception()

@cli.command()
@click.option("--input-dir", required=True, help="Directory containing audio files to augment")
@click.option("--output-dir", required=True, help="Directory to save augmented files")
def augment(input_dir: str, output_dir: str):
    """
    Applies augmentation effects (noise, reverb) to existing audio files.
    """
    console.print("[yellow]Augmentation command not yet fully implemented.[/yellow]")
    console.print(f"Would augment files from {input_dir} to {output_dir}")
    # TODO: Connect to wakegen.augmentation.pipeline

@cli.command()
@click.option("--data-dir", required=True, help="Directory containing the dataset")
def validate(data_dir: str):
    """
    Runs quality assurance checks on the dataset.
    """
    console.print("[yellow]Validation command not yet fully implemented.[/yellow]")
    console.print(f"Would validate dataset in {data_dir}")
    # TODO: Connect to wakegen.quality.validator

@cli.command()
@click.option("--data-dir", required=True, help="Directory containing the dataset")
@click.option("--format", default="openwakeword", help="Export format (default: openwakeword)")
@click.option("--output-path", required=True, help="Path to save the exported manifest/files")
def export(data_dir: str, format: str, output_path: str):
    """
    Exports the dataset to a specific format for training.
    """
    console.print("[yellow]Export command not yet fully implemented.[/yellow]")
    console.print(f"Would export {data_dir} as {format} to {output_path}")
    # TODO: Connect to wakegen.export

@cli.command()
@click.option("--model-type", default="openwakeword", help="Type of model to train")
@click.option("--output-script", default="train.sh", help="Path to save the training script")
def train_script(model_type: str, output_script: str):
    """
    Generates a training script for the selected model type.
    """
    console.print("[yellow]Training script generation not yet fully implemented.[/yellow]")
    console.print(f"Would generate {model_type} training script at {output_script}")
    # TODO: Connect to wakegen.training.script_generator


# =============================================================================
# BATCH COMMAND
# =============================================================================
# Process multiple wake words from a file or configuration.


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True),
    help="File containing wake words (one per line) or a wakegen.yaml config"
)
@click.option("--count", "-n", default=100, help="Number of samples to generate per wake word")
@click.option("--output-dir", "-o", default="./output", help="Directory to save generated files")
@click.option("--provider", "-p", default="edge_tts", help="TTS provider to use")
@click.option("--voice", "-v", help="Specific voice ID to use")
@click.option(
    "--split-by-provider",
    is_flag=True,
    help="Use multiple providers and distribute samples among them"
)
@click.option(
    "--providers",
    multiple=True,
    help="List of providers to use when --split-by-provider is set (e.g., --providers edge_tts --providers piper)"
)
def batch(
    input_file: Optional[str],
    count: int,
    output_dir: str,
    provider: str,
    voice: Optional[str],
    split_by_provider: bool,
    providers: tuple
):
    """
    Generate samples for multiple wake words in batch mode.
    
    This command is optimized for generating large datasets with multiple wake words.
    You can provide wake words via:
    
    1. A text file (one wake word per line)
    2. A wakegen.yaml configuration file (uses generation.wake_words)
    
    If no input file is provided, it will prompt for wake words interactively.
    
    Examples:
        # From a text file
        wakegen batch --input words.txt --count 100 --output-dir ./dataset
        
        # From a config file
        wakegen batch --input wakegen.yaml
        
        # With multiple providers
        wakegen batch --input words.txt --split-by-provider --providers edge_tts --providers piper
    """
    asyncio.run(run_batch_generation(
        input_file=input_file,
        count=count,
        output_dir=output_dir,
        provider_name=provider,
        voice_id=voice,
        split_by_provider=split_by_provider,
        provider_list=list(providers) if providers else None
    ))


async def run_batch_generation(
    input_file: Optional[str],
    count: int,
    output_dir: str,
    provider_name: str,
    voice_id: Optional[str],
    split_by_provider: bool,
    provider_list: Optional[list]
):
    """
    Async implementation of batch generation.
    
    This handles the logic for reading wake words from various sources
    and distributing generation across providers.
    """
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    
    wake_words: list[str] = []
    
    # 1. Load wake words from input source
    if input_file:
        input_path = Path(input_file)
        
        # Check if it's a YAML config file
        if input_path.suffix.lower() in (".yaml", ".yml"):
            try:
                config = load_config(input_path)
                wake_words = config.generation.wake_words
                # Also use config's output_dir and count if not overridden
                if output_dir == "./output":
                    output_dir = config.generation.output_dir
                if count == 100:
                    count = config.generation.count
                console.print(f"[green]Loaded {len(wake_words)} wake words from config[/green]")
            except ConfigError as e:
                console.print(f"[bold red]Error loading config:[/bold red] {e}")
                return
        else:
            # It's a text file - read one wake word per line
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                wake_words = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
                console.print(f"[green]Loaded {len(wake_words)} wake words from text file[/green]")
            except Exception as e:
                console.print(f"[bold red]Error reading file:[/bold red] {e}")
                return
    else:
        # Interactive mode - prompt for wake words
        console.print("[bold cyan]Enter wake words (one per line, empty line to finish):[/bold cyan]")
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    break
                wake_words.append(line)
            except (EOFError, KeyboardInterrupt):
                break
    
    if not wake_words:
        console.print("[yellow]No wake words provided. Exiting.[/yellow]")
        return
    
    # 2. Determine providers to use
    providers_to_use: list[tuple[ProviderType, float]] = []
    
    if split_by_provider:
        # Use multiple providers with equal weights
        if provider_list:
            for p_name in provider_list:
                try:
                    p_type = ProviderType(p_name.lower())
                    providers_to_use.append((p_type, 1.0 / len(provider_list)))
                except ValueError:
                    console.print(f"[yellow]Warning: Unknown provider '{p_name}', skipping[/yellow]")
        else:
            # Default to a few common providers
            default_providers = [ProviderType.EDGE_TTS]
            for p_type in default_providers:
                providers_to_use.append((p_type, 1.0 / len(default_providers)))
    else:
        # Single provider
        try:
            p_type = ProviderType(provider_name.lower())
            providers_to_use.append((p_type, 1.0))
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider_name}'")
            return
    
    # 3. Display generation plan
    total_samples = len(wake_words) * count
    console.print(Panel(
        f"[bold]Wake Words:[/bold] {len(wake_words)}\n"
        f"[bold]Samples per word:[/bold] {count}\n"
        f"[bold]Total samples:[/bold] {total_samples}\n"
        f"[bold]Output:[/bold] {output_dir}\n"
        f"[bold]Providers:[/bold] {', '.join(p[0].value for p in providers_to_use)}",
        title="Batch Generation Plan",
        border_style="cyan"
    ))
    
    # 4. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Initialize providers and get voices
    provider_config = get_provider_config()
    provider_instances: dict[ProviderType, tuple[TTSProvider, Optional[any]]] = {}
    
    for p_type, _ in providers_to_use:
        try:
            provider = get_provider(p_type, provider_config)
            
            # Get voice to use
            selected_voice = None
            if voice_id:
                from wakegen.core.protocols import Voice
                from wakegen.core.types import Gender
                selected_voice = Voice(id=voice_id, name=voice_id, language="unknown", gender=Gender.NEUTRAL)
            else:
                voices = await provider.list_voices()
                if voices:
                    selected_voice = next(
                        (v for v in voices if v.language.startswith("en-")),
                        voices[0]
                    )
            
            provider_instances[p_type] = (provider, selected_voice)
            console.print(f"  [green]✓[/green] {p_type.value}: {selected_voice.name if selected_voice else 'no voice'}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {p_type.value}: {e}")
    
    if not provider_instances:
        console.print("[bold red]No providers available. Exiting.[/bold red]")
        return
    
    # 6. Generate samples with progress tracking
    total_generated = 0
    total_failed = 0
    
    # Use Rich progress for nice display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        overall_task = progress.add_task("[bold]Overall Progress", total=total_samples)
        
        for wake_word in wake_words:
            word_dir = os.path.join(output_dir, wake_word.replace(" ", "_").lower())
            os.makedirs(word_dir, exist_ok=True)
            
            word_task = progress.add_task(f"  '{wake_word}'", total=count)
            sample_index = 0
            
            # Distribute samples across providers
            for p_type, weight in providers_to_use:
                if p_type not in provider_instances:
                    continue
                
                provider, voice = provider_instances[p_type]
                if not voice:
                    continue
                
                provider_count = int(count * weight)
                
                for _ in range(provider_count):
                    sample_index += 1
                    filename = f"{wake_word.replace(' ', '_').lower()}_{sample_index:04d}_{p_type.value}.wav"
                    file_path = os.path.join(word_dir, filename)
                    
                    try:
                        await provider.generate(wake_word, voice.id, file_path)
                        total_generated += 1
                    except Exception as e:
                        total_failed += 1
                        # Log but don't stop
                        pass
                    
                    progress.update(word_task, advance=1)
                    progress.update(overall_task, advance=1)
            
            progress.remove_task(word_task)
    
    # 7. Summary
    console.print(Panel(
        f"[bold green]✓ Batch generation complete![/bold green]\n\n"
        f"[bold]Generated:[/bold] {total_generated} samples\n"
        f"[bold]Failed:[/bold] {total_failed}\n"
        f"[bold]Wake words processed:[/bold] {len(wake_words)}\n"
        f"[bold]Output directory:[/bold] {output_dir}",
        title="Summary",
        border_style="green"
    ))


# =============================================================================
# CONFIG COMMAND GROUP
# =============================================================================
# These commands help you manage wakegen.yaml configuration files.
# Think of it like a setup wizard - init creates a starter config,
# validate checks if your config is correct before running.


@cli.group()
def config():
    """
    Manage wakegen configuration files.
    
    Use these commands to create and validate wakegen.yaml configuration files.
    A config file lets you define all generation settings in one place instead
    of passing many command-line arguments.
    
    Examples:
        wakegen config init              # Create a starter config file
        wakegen config validate my.yaml  # Check if a config file is valid
    """
    pass


@config.command(name="init")
@click.option(
    "--output",
    "-o",
    default="wakegen.yaml",
    help="Output file path (default: wakegen.yaml in current directory)"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing file without asking"
)
def config_init(output: str, force: bool):
    """
    Generate a template wakegen.yaml configuration file.
    
    Creates a well-commented starter configuration with sensible defaults.
    Edit the generated file to customize your wake word generation settings.
    
    The template includes:
    - Project metadata (name, version)
    - Generation settings (wake words, count, output)
    - Provider configuration (TTS services)
    - Augmentation profiles (noise, reverb)
    - Export settings (train/val/test splits)
    
    Examples:
        wakegen config init                    # Create wakegen.yaml
        wakegen config init -o my_project.yaml # Custom filename
        wakegen config init --force            # Overwrite existing
    """
    output_path = Path(output)
    
    # Check if file already exists
    if output_path.exists() and not force:
        console.print(f"[yellow]File already exists:[/yellow] {output_path}")
        console.print("Use [cyan]--force[/cyan] to overwrite or choose a different name with [cyan]--output[/cyan].")
        raise SystemExit(1)
    
    # Get the template content
    template = get_template_config()
    
    try:
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the template
        output_path.write_text(template, encoding="utf-8")
        
        # Success message with next steps
        console.print(Panel(
            f"[bold green]✓ Created configuration file:[/bold green] {output_path}\n\n"
            "[dim]Next steps:[/dim]\n"
            f"  1. Edit [cyan]{output_path}[/cyan] to customize your settings\n"
            f"  2. Run [cyan]wakegen config validate {output_path}[/cyan] to check for errors\n"
            f"  3. Run [cyan]wakegen generate --config {output_path}[/cyan] to start generating",
            title="Configuration Created",
            border_style="green"
        ))
        
    except PermissionError:
        console.print(f"[bold red]Error:[/bold red] Permission denied writing to: {output_path}")
        console.print("Try a different location or check your permissions.")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to create configuration file: {e}")
        raise SystemExit(1)


@config.command(name="validate")
@click.argument("path", type=click.Path(exists=False))
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed configuration values"
)
def config_validate(path: str, verbose: bool):
    """
    Validate a wakegen configuration file.
    
    Checks your YAML configuration for:
    - Syntax errors (missing colons, bad indentation)
    - Missing required fields (project name, wake words)
    - Invalid values (negative counts, bad ratios)
    - Unknown fields (catches typos)
    - Environment variable resolution
    
    Exit code:
    - 0: Configuration is valid
    - 1: Configuration has errors
    
    Examples:
        wakegen config validate wakegen.yaml
        wakegen config validate wakegen.yaml --verbose
    """
    config_path = Path(path)
    
    # Check file exists first (more helpful error message)
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {config_path}")
        console.print("\nTo create a new configuration file, run:")
        console.print(f"  [cyan]wakegen config init --output {config_path}[/cyan]")
        raise SystemExit(1)
    
    # Try to load and validate the configuration
    try:
        config_obj = load_config(config_path)
        
        # Success! Show validation passed
        console.print(Panel(
            f"[bold green]✓ Configuration is valid![/bold green]\n\n"
            f"[dim]File:[/dim] {config_path}",
            title="Validation Passed",
            border_style="green"
        ))
        
        # If verbose, show the parsed configuration
        if verbose:
            _print_config_summary(config_obj)
        
    except ConfigError as e:
        # Show validation error with helpful message
        console.print(Panel(
            f"[bold red]✗ Configuration validation failed[/bold red]\n\n"
            f"[dim]File:[/dim] {config_path}\n\n"
            f"{e}",
            title="Validation Error",
            border_style="red"
        ))
        raise SystemExit(1)
    except Exception as e:
        # Unexpected error
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        raise SystemExit(1)


def _print_config_summary(config: WakegenConfig) -> None:
    """
    Print a human-readable summary of the configuration.
    
    This helper function displays the validated configuration in a nice format,
    useful for debugging and verification.
    
    Args:
        config: Validated WakegenConfig object to display.
    """
    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]\n")
    
    # Project section
    console.print("[bold]Project:[/bold]")
    console.print(f"  Name: [green]{config.project.name}[/green]")
    console.print(f"  Version: {config.project.version}")
    if config.project.description:
        console.print(f"  Description: {config.project.description}")
    
    # Generation section
    console.print("\n[bold]Generation:[/bold]")
    console.print(f"  Wake Words: {config.generation.wake_words}")
    console.print(f"  Count per word: {config.generation.count}")
    console.print(f"  Output: {config.generation.output_dir}")
    console.print(f"  Format: {config.generation.audio_format} @ {config.generation.sample_rate}Hz")
    
    # Providers section
    console.print("\n[bold]Providers:[/bold]")
    for p in config.providers:
        voices_str = ", ".join(p.voices[:3]) if p.voices else "default"
        if len(p.voices) > 3:
            voices_str += f" (+{len(p.voices) - 3} more)"
        console.print(f"  • [cyan]{p.type}[/cyan] (weight: {p.weight:.1%}) - {voices_str}")
    
    # Augmentation section
    console.print("\n[bold]Augmentation:[/bold]")
    console.print(f"  Enabled: {'[green]yes[/green]' if config.augmentation.enabled else '[red]no[/red]'}")
    if config.augmentation.enabled:
        console.print(f"  Profiles: {config.augmentation.profiles}")
        console.print(f"  Copies per original: {config.augmentation.augmented_per_original}")
    
    # Export section
    console.print("\n[bold]Export:[/bold]")
    console.print(f"  Format: {config.export.format}")
    train, val, test = config.export.split_ratio
    console.print(f"  Split: train {train:.0%} / val {val:.0%} / test {test:.0%}")
    
    console.print("")  # Final newline


# =============================================================================
# PLUGIN COMMAND GROUP
# =============================================================================
# These commands help you manage third-party TTS provider plugins.
# Plugins extend wakegen with new TTS providers without modifying the core code.


@cli.group()
def plugin():
    """
    Manage third-party TTS provider plugins.
    
    Plugins allow the community to add new TTS providers to wakegen.
    Install plugins via pip and they'll be automatically discovered.
    
    Examples:
        wakegen plugin list              # Show all installed plugins
        wakegen plugin info my-plugin    # Details about a specific plugin
        wakegen plugin reload            # Rescan for new plugins
    """
    pass


@plugin.command(name="list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed plugin information")
def plugin_list(verbose: bool):
    """
    List all installed wakegen plugins.
    
    Shows all discovered TTS provider plugins with their status.
    Plugins are auto-discovered from installed Python packages that
    register the 'wakegen.plugins' entry point.
    
    Examples:
        wakegen plugin list
        wakegen plugin list --verbose
    """
    from wakegen.plugins import discover_plugins, PluginLoadError
    
    try:
        plugins = discover_plugins()
    except Exception as e:
        console.print(f"[bold red]Error discovering plugins:[/bold red] {e}")
        return
    
    if not plugins:
        console.print(Panel(
            "[yellow]No plugins installed.[/yellow]\n\n"
            "Plugins extend wakegen with additional TTS providers.\n"
            "Install plugins via pip:\n"
            "  [cyan]pip install wakegen-plugin-example[/cyan]\n\n"
            "Or create your own! See the documentation for details.",
            title="No Plugins Found",
            border_style="yellow"
        ))
        return
    
    # Create table
    table = Table(title="Installed Plugins", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Description")
    
    if verbose:
        table.add_column("Author")
        table.add_column("Requirements")
    
    for p in plugins:
        # Status indicator
        if p.is_enabled:
            status = "[green]✓ Enabled[/green]"
        elif p.load_error:
            status = f"[red]✗ Error[/red]"
        else:
            status = "[yellow]○ Disabled[/yellow]"
        
        if verbose:
            # Build requirements string
            reqs = []
            if p.metadata.requires_api_key:
                reqs.append("API Key")
            if p.metadata.requires_gpu:
                reqs.append("GPU")
            req_str = ", ".join(reqs) if reqs else "[dim]None[/dim]"
            
            table.add_row(
                p.name,
                p.metadata.version,
                status,
                p.metadata.description[:50] + "..." if len(p.metadata.description) > 50 else p.metadata.description,
                p.metadata.author,
                req_str
            )
        else:
            table.add_row(
                p.name,
                p.metadata.version,
                status,
                p.metadata.description[:60] + "..." if len(p.metadata.description) > 60 else p.metadata.description
            )
    
    console.print(table)
    console.print(f"\n[dim]Found {len(plugins)} plugin(s)[/dim]")


@plugin.command(name="info")
@click.argument("name")
def plugin_info(name: str):
    """
    Show detailed information about a specific plugin.
    
    Displays the plugin's metadata, capabilities, supported languages,
    and usage instructions.
    
    Examples:
        wakegen plugin info my-tts-plugin
    """
    from wakegen.plugins import discover_plugins, get_plugin
    
    # Ensure plugins are discovered
    discover_plugins()
    
    plugin = get_plugin(name)
    
    if not plugin:
        console.print(f"[bold red]Plugin not found:[/bold red] {name}")
        console.print("\nUse [cyan]wakegen plugin list[/cyan] to see installed plugins.")
        raise SystemExit(1)
    
    meta = plugin.metadata
    
    # Build info panel
    info_lines = [
        f"[bold]Name:[/bold] {meta.name}",
        f"[bold]Version:[/bold] {meta.version}",
        f"[bold]Author:[/bold] {meta.author}",
        f"[bold]Description:[/bold] {meta.description}",
        "",
        f"[bold]Status:[/bold] {'[green]Enabled[/green]' if plugin.is_enabled else '[red]Disabled[/red]'}",
        f"[bold]Entry Point:[/bold] {plugin.entry_point}",
        "",
        "[bold]Requirements:[/bold]",
        f"  API Key: {'[yellow]Yes[/yellow]' if meta.requires_api_key else '[green]No[/green]'}",
        f"  GPU: {'[yellow]Recommended[/yellow]' if meta.requires_gpu else '[green]Not required[/green]'}",
        "",
        f"[bold]Supported Languages:[/bold] {', '.join(meta.supported_languages)}",
    ]
    
    if meta.homepage:
        info_lines.append(f"\n[bold]Homepage:[/bold] {meta.homepage}")
    
    if plugin.load_error:
        info_lines.append(f"\n[bold red]Load Error:[/bold red] {plugin.load_error}")
    
    console.print(Panel(
        "\n".join(info_lines),
        title=f"Plugin: {meta.name}",
        border_style="cyan"
    ))
    
    # Usage example
    console.print("\n[bold]Usage:[/bold]")
    console.print(f"  wakegen generate --provider plugin:{meta.name} --text \"hello\"")


@plugin.command(name="reload")
def plugin_reload():
    """
    Reload all plugins.
    
    Rescans for installed plugins without restarting wakegen.
    Useful after installing new plugins via pip.
    
    Examples:
        pip install wakegen-plugin-new
        wakegen plugin reload
        wakegen plugin list  # Now shows the new plugin
    """
    from wakegen.plugins import reload_plugins
    
    console.print("[dim]Reloading plugins...[/dim]")
    
    try:
        plugins = reload_plugins()
        console.print(f"[bold green]✓ Reloaded {len(plugins)} plugin(s)[/bold green]")
        
        if plugins:
            for p in plugins:
                status = "[green]✓[/green]" if p.is_enabled else "[red]✗[/red]"
                console.print(f"  {status} {p.name} v{p.metadata.version}")
        else:
            console.print("[dim]No plugins found.[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Error reloading plugins:[/bold red] {e}")
        raise SystemExit(1)


@plugin.command(name="create")
@click.argument("name")
@click.option("--output-dir", "-o", default=".", help="Directory to create plugin in")
def plugin_create(name: str, output_dir: str):
    """
    Create a template for a new wakegen plugin.
    
    Generates a starter plugin package with all the boilerplate code needed
    to create a new TTS provider plugin.
    
    Examples:
        wakegen plugin create my-tts-plugin
        wakegen plugin create my-tts-plugin --output-dir ./plugins
    """
    import os
    
    # Normalize name
    plugin_name = name.lower().replace("_", "-").replace(" ", "-")
    module_name = plugin_name.replace("-", "_")
    class_name = "".join(word.capitalize() for word in plugin_name.split("-")) + "Plugin"
    
    # Create directory structure
    plugin_dir = os.path.join(output_dir, plugin_name)
    src_dir = os.path.join(plugin_dir, module_name)
    
    if os.path.exists(plugin_dir):
        console.print(f"[bold red]Directory already exists:[/bold red] {plugin_dir}")
        raise SystemExit(1)
    
    os.makedirs(src_dir)
    
    # Create pyproject.toml
    pyproject_content = f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{plugin_name}"
version = "0.1.0"
description = "A wakegen TTS provider plugin"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "wakegen",
]

[project.entry-points."wakegen.plugins"]
{plugin_name} = "{module_name}.provider:{class_name}"
'''
    
    with open(os.path.join(plugin_dir, "pyproject.toml"), "w") as f:
        f.write(pyproject_content)
    
    # Create README.md
    readme_content = f'''# {plugin_name}

A wakegen TTS provider plugin.

## Installation

```bash
pip install {plugin_name}
```

## Usage

```bash
wakegen generate --provider plugin:{plugin_name} --text "hello"
```
'''
    
    with open(os.path.join(plugin_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Create __init__.py
    init_content = f'''"""
{plugin_name} - A wakegen TTS provider plugin.
"""

from {module_name}.provider import {class_name}

__all__ = ["{class_name}"]
'''
    
    with open(os.path.join(src_dir, "__init__.py"), "w") as f:
        f.write(init_content)
    
    # Create provider.py
    provider_content = f'''"""
{class_name} - TTS Provider Implementation

This is the main plugin implementation. Modify the generate() and list_voices()
methods to integrate with your TTS service.
"""

from typing import List
from wakegen.plugins import TTSPlugin, PluginMetadata
from wakegen.core.protocols import Voice
from wakegen.core.types import Gender


class {class_name}(TTSPlugin):
    """
    {plugin_name} TTS Provider.
    
    Replace this with a description of your TTS provider.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{plugin_name}",
            version="0.1.0",
            description="A custom TTS provider for wakegen",
            author="Your Name",
            homepage="https://github.com/yourusername/{plugin_name}",
            requires_api_key=False,
            requires_gpu=False,
            supported_languages=["en"],
        )
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio from text.
        
        Implement your TTS logic here. This method should:
        1. Call your TTS API/library with the text and voice_id
        2. Save the resulting audio to output_path
        
        Args:
            text: The text to convert to speech.
            voice_id: The voice ID from list_voices().
            output_path: Where to save the audio file.
        """
        # TODO: Implement your TTS generation logic
        raise NotImplementedError("Implement your TTS generation here")
    
    async def list_voices(self) -> List[Voice]:
        """
        List available voices.
        
        Return a list of Voice objects that can be used with generate().
        
        Returns:
            List of available voices.
        """
        # TODO: Return your available voices
        return [
            Voice(
                id="default",
                name="Default Voice",
                language="en-US",
                gender=Gender.NEUTRAL,
            ),
        ]
    
    async def validate_config(self) -> None:
        """
        Validate plugin configuration.
        
        Check that all required dependencies, API keys, etc. are available.
        Raise PluginValidationError if something is wrong.
        """
        # TODO: Add any validation checks
        pass
'''
    
    with open(os.path.join(src_dir, "provider.py"), "w") as f:
        f.write(provider_content)
    
    console.print(Panel(
        f"[bold green]✓ Created plugin template![/bold green]\n\n"
        f"[bold]Directory:[/bold] {plugin_dir}\n"
        f"[bold]Module:[/bold] {module_name}\n"
        f"[bold]Class:[/bold] {class_name}\n\n"
        "[dim]Next steps:[/dim]\n"
        f"  1. cd {plugin_dir}\n"
        f"  2. Edit {module_name}/provider.py to implement your TTS logic\n"
        "  3. pip install -e .  # Install in development mode\n"
        "  4. wakegen plugin list  # Verify it's discovered",
        title="Plugin Created",
        border_style="green"
    ))


# =============================================================================
# CACHE COMMAND GROUP
# =============================================================================
# These commands help you manage the generation cache.
# Caching avoids regenerating the same audio files, saving time and API costs.


@cli.group()
def cache():
    """
    Manage the audio generation cache.
    
    Wakegen caches generated audio files to avoid redundant TTS calls.
    This saves time and API costs when regenerating similar datasets.
    
    Examples:
        wakegen cache stats    # Show cache statistics
        wakegen cache clear    # Remove all cached files
        wakegen cache path     # Show cache directory location
    """
    pass


@cache.command(name="stats")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def cache_stats(as_json: bool):
    """
    Show cache statistics.
    
    Displays information about the cache including:
    - Number of cached files
    - Total cache size
    - Hit/miss ratio
    - Number of evictions
    
    Examples:
        wakegen cache stats
        wakegen cache stats --json
    """
    from wakegen.utils.caching import GenerationCache
    import json
    
    cache = GenerationCache()
    stats = cache.get_stats()
    
    if as_json:
        console.print(json.dumps(stats.to_dict(), indent=2))
    else:
        console.print(Panel(
            f"[bold]Cache Statistics[/bold]\n\n"
            f"[bold]Files:[/bold] {stats.file_count}\n"
            f"[bold]Size:[/bold] {stats.total_size_mb:.2f} MB\n"
            f"[bold]Hits:[/bold] {stats.hits}\n"
            f"[bold]Misses:[/bold] {stats.misses}\n"
            f"[bold]Hit Rate:[/bold] {stats.hit_rate:.1%}\n"
            f"[bold]Evictions:[/bold] {stats.evictions}",
            title="Cache Stats",
            border_style="cyan"
        ))


@cache.command(name="clear")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def cache_clear(force: bool):
    """
    Clear all cached files.
    
    This removes all cached audio files. Use this if you want to
    force regeneration of all samples or free up disk space.
    
    Examples:
        wakegen cache clear
        wakegen cache clear --force
    """
    from wakegen.utils.caching import GenerationCache
    
    cache = GenerationCache()
    stats = cache.get_stats()
    
    if stats.file_count == 0:
        console.print("[yellow]Cache is already empty.[/yellow]")
        return
    
    if not force:
        console.print(f"[yellow]Warning:[/yellow] This will delete {stats.file_count} cached files ({stats.total_size_mb:.2f} MB).")
        if not click.confirm("Continue?"):
            console.print("[dim]Cancelled.[/dim]")
            return
    
    cache.clear()
    console.print("[bold green]✓ Cache cleared successfully.[/bold green]")


@cache.command(name="path")
def cache_path():
    """
    Show the cache directory location.
    
    Use this to find where cached files are stored, for example
    to back them up or examine individual files.
    
    Examples:
        wakegen cache path
    """
    from wakegen.utils.caching import GenerationCache
    
    cache = GenerationCache()
    console.print(f"[bold]Cache directory:[/bold] {cache.cache_dir.absolute()}")
    
    if cache.cache_dir.exists():
        console.print(f"[dim]Directory exists with {len(list(cache.cache_dir.glob('*')))} files[/dim]")
    else:
        console.print("[dim]Directory does not exist yet (will be created on first use)[/dim]")


@cache.command(name="list")
@click.option("--limit", "-n", default=20, help="Maximum entries to show")
@click.option("--sort", "-s", type=click.Choice(["recent", "oldest", "size"]), default="recent", help="Sort order")
def cache_list(limit: int, sort: str):
    """
    List cached files.
    
    Shows the most recently cached files with their metadata.
    
    Examples:
        wakegen cache list
        wakegen cache list --limit 50
        wakegen cache list --sort size
    """
    from wakegen.utils.caching import GenerationCache
    
    cache = GenerationCache()
    entries = list(cache._entries.values())
    
    if not entries:
        console.print("[yellow]Cache is empty.[/yellow]")
        return
    
    # Sort entries
    if sort == "recent":
        entries.sort(key=lambda e: e.last_accessed, reverse=True)
    elif sort == "oldest":
        entries.sort(key=lambda e: e.last_accessed)
    elif sort == "size":
        entries.sort(key=lambda e: e.file_size, reverse=True)
    
    # Limit entries
    entries = entries[:limit]
    
    # Display as table
    table = Table(title=f"Cached Files (showing {len(entries)} of {cache._stats.file_count})", show_header=True, header_style="bold cyan")
    table.add_column("Text", max_width=30)
    table.add_column("Provider")
    table.add_column("Voice")
    table.add_column("Size")
    table.add_column("Accesses")
    
    for entry in entries:
        text_preview = entry.text[:27] + "..." if len(entry.text) > 30 else entry.text
        size_str = f"{entry.file_size / 1024:.1f} KB"
        table.add_row(
            text_preview,
            entry.provider,
            entry.voice_id[:15] + "..." if len(entry.voice_id) > 15 else entry.voice_id,
            size_str,
            str(entry.access_count)
        )
    
    console.print(table)


# =============================================================================
# GPU STATUS COMMAND
# =============================================================================
# This command helps you check GPU availability for TTS models.


@cli.command(name="gpu-status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def gpu_status(as_json: bool):
    """
    Show GPU status and availability.
    
    Displays information about available GPUs including:
    - Number of GPUs
    - GPU names and memory
    - Backend (CUDA, MPS, or CPU)
    
    This helps you understand which TTS providers will run on GPU
    vs CPU for your system.
    
    Examples:
        wakegen gpu-status
        wakegen gpu-status --json
    """
    from wakegen.utils.gpu import detect_gpu_status, GPUBackend
    import json
    
    status = detect_gpu_status()
    
    if as_json:
        console.print(json.dumps(status.to_dict(), indent=2))
        return
    
    # Display GPU information
    if not status.has_gpu:
        console.print(Panel(
            "[yellow]No GPU detected.[/yellow]\n\n"
            "TTS models will run on CPU.\n"
            "For faster generation with GPU-based models (Coqui XTTS, StyleTTS2, etc.),\n"
            "consider using a system with NVIDIA GPU and PyTorch CUDA support.",
            title="GPU Status",
            border_style="yellow"
        ))
        return
    
    # Build GPU table
    table = Table(title=f"GPU Status ({status.backend.value.upper()})", show_header=True, header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Total Memory")
    table.add_column("Free Memory")
    table.add_column("Utilization")
    table.add_column("Status")
    
    for gpu in status.gpus:
        utilization = f"{gpu.utilization:.1%}"
        status_str = "[green]✓ Available[/green]" if gpu.is_available else "[yellow]⚠ Low Memory[/yellow]"
        table.add_row(
            str(gpu.id),
            gpu.name,
            f"{gpu.total_memory_mb:.0f} MB",
            f"{gpu.free_memory_mb:.0f} MB",
            utilization,
            status_str
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Total GPUs:[/bold] {status.num_gpus}")
    console.print(f"[bold]Total Memory:[/bold] {status.total_memory_mb:.0f} MB")
    console.print(f"[bold]Total Free:[/bold] {status.total_free_mb:.0f} MB")
    
    # Provider recommendations
    console.print("\n[bold]Provider GPU Support:[/bold]")
    console.print("  [green]✓[/green] coqui_xtts - GPU recommended (large model)")
    console.print("  [green]✓[/green] kokoro - CPU-friendly (82M params)")
    console.print("  [green]✓[/green] piper - CPU-friendly (lightweight)")
    console.print("  [green]✓[/green] mimic3 - CPU-friendly (lightweight)")
    console.print("  [dim]○[/dim] edge_tts - Cloud-based (no local GPU)")