"""
Generate Commands Module

This module contains CLI commands for generating audio samples.
These are the core commands for creating wake word datasets.

Commands:
- generate: Generate audio samples with wizard, CLI args, or config file
- wizard: Interactive wizard for generation
"""

import click
import asyncio
import os
from pathlib import Path
from typing import Optional, List

from rich.console import Console
from rich.progress import track
from rich.panel import Panel

from wakegen.config.settings import get_generation_config, get_provider_config
from wakegen.config.yaml_loader import load_config, WakegenConfig
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
from wakegen.core.exceptions import ConfigError
from wakegen.utils.audio import resample_audio
from wakegen.ui.cli.wizard import run_wizard


# =============================================================================
# SHARED CONSOLE
# =============================================================================

console = Console()


# =============================================================================
# GENERATE COMMAND
# =============================================================================


def register_generate_commands(cli_group: click.Group) -> None:
    """
    Register generation-related commands with the main CLI group.
    
    Args:
        cli_group: The main Click group to add commands to.
    """
    
    @cli_group.command()
    @click.option("--text", "-t", help="The wake word or phrase to generate")
    @click.option("--count", "-n", default=10, help="Number of samples to generate")
    @click.option("--output-dir", "-o", default="./output", help="Directory to save generated files")
    @click.option("--provider", "-p", default="edge_tts", help="TTS provider to use (default: edge_tts)")
    @click.option("--voice", "-v", help="Specific voice ID to use (optional)")
    @click.option("--config", "-c", type=click.Path(exists=True), help="Path to a wakegen.yaml config file")
    @click.option("--language", "-l", multiple=True, help="Filter voices by language code (e.g., tr-TR)")
    def generate(
        text: Optional[str],
        count: int,
        output_dir: str,
        provider: str,
        voice: Optional[str],
        config: Optional[str],
        language: tuple
    ) -> None:
        """
        Generates audio samples for a wake word.
        
        This is the main command! You can use it in three ways:
        1. Interactive Wizard: Run 'wakegen generate' without arguments
        2. Command Line: Provide arguments like --text "hey computer"
        3. Config File: Use --config wakegen.yaml for reproducible runs
        
        Examples:
            wakegen generate  # Starts wizard
            wakegen generate --text "hello world" --count 5
            wakegen generate --config my_project.yaml
            wakegen generate --text "merhaba" --language tr-TR
        """
        # Mode 1: Config file
        if config:
            asyncio.run(_run_generation_from_config(config, language))
            return

        # Mode 2: Command line arguments
        if text:
            asyncio.run(_run_generation(
                text=text,
                count=count,
                output_dir=output_dir,
                provider_name=provider,
                voice_id=voice,
                languages=list(language) if language else None
            ))
            return

        # Mode 3: Interactive Wizard
        asyncio.run(_run_interactive_generation())


    @cli_group.command()
    def wizard() -> None:
        """
        Interactive wizard for generating wake word samples.
        
        This launches a step-by-step guide that helps you:
        - Choose your wake word(s)
        - Select TTS providers and voices
        - Configure sample count and output location
        - Start generation with sensible defaults
        
        Perfect for new users or quick one-off generations!
        
        Example:
            wakegen wizard
        """
        asyncio.run(_run_interactive_generation())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_interactive_generation() -> None:
    """
    Runs the generation process using the interactive wizard.
    """
    config = await run_wizard()
    
    if not config:
        return

    # Map wizard config to generation arguments
    await _run_generation(
        text=config["wake_word"],
        count=config["count"],
        preset=None,
        output_dir=config["output_dir"],
        provider_name="edge_tts",
        voice_id=None
    )


async def _run_generation_from_config(
    config_path: str,
    language_override: Optional[tuple] = None
) -> None:
    """
    Runs generation based on a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        language_override: Optional tuple of languages to override config.
    """
    try:
        # Load configuration
        config = load_config(Path(config_path))
        
        console.print(Panel(
            f"[bold]Project:[/bold] {config.project.name} v{config.project.version}\n"
            f"[bold]Wake Words:[/bold] {config.generation.wake_words}\n"
            f"[bold]Providers:[/bold] {len(config.providers)} configured",
            title="Starting Generation from Config",
            border_style="green"
        ))
        
        # Use languages from config, or override if provided
        languages = list(language_override) if language_override else None
        
        # Iterate over providers in config
        for p_config in config.providers:
            try:
                # Initialize provider
                provider = get_provider(p_config.type, get_provider_config())
                
                # Determine voices to use
                voices_to_use = []
                if p_config.voices:
                    all_voices = await provider.list_voices()
                    voices_to_use = [v for v in all_voices if v.id in p_config.voices or v.name in p_config.voices]
                else:
                    all_voices = await provider.list_voices()
                    filter_langs = languages or p_config.languages
                    if filter_langs:
                        all_voices = [v for v in all_voices if any(l.lower() in v.language.lower() for l in filter_langs)]
                    
                    if not all_voices:
                        console.print(f"[yellow]No voices found for {p_config.type} matching criteria. Skipping.[/yellow]")
                        continue
                    
                    voices_to_use = all_voices[:1]
                
                if not voices_to_use:
                    console.print(f"[yellow]No valid voices found for {p_config.type}. Skipping.[/yellow]")
                    continue
                
                # Generate for each wake word and each voice
                for wake_word in config.generation.wake_words:
                    for voice in voices_to_use:
                        console.print(f"[bold]Generating:[/bold] '{wake_word}' with {p_config.type} ({voice.name})")
                        
                        count = int(config.generation.count * p_config.weight)
                        if count < 1:
                            count = 1
                        
                        word_dir = os.path.join(config.generation.output_dir, wake_word.replace(" ", "_").lower())
                        
                        await _run_generation(
                            text=wake_word,
                            count=count,
                            output_dir=word_dir,
                            provider_name=p_config.type,
                            voice_id=voice.id,
                            languages=None
                        )
                        
            except Exception as e:
                console.print(f"[red]Error with provider {p_config.type}: {e}[/red]")
                
    except ConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


async def _run_generation(
    text: str,
    count: int,
    output_dir: str,
    provider_name: str,
    voice_id: Optional[str] = None,
    preset: Optional[str] = None,
    languages: Optional[List[str]] = None
) -> None:
    """
    Core generation logic.
    
    Args:
        text: Wake word or phrase to generate.
        count: Number of samples to generate.
        output_dir: Directory to save generated files.
        provider_name: Name of the TTS provider.
        voice_id: Optional specific voice ID.
        preset: Optional preset name (unused currently).
        languages: Optional list of language codes to filter voices.
    """
    try:
        # 1. Setup Provider
        p_type = ProviderType(provider_name.lower())
        provider_config = get_provider_config()
        provider = get_provider(p_type, provider_config)
        
        # 2. Get Voice
        selected_voice = None
        if voice_id:
            voices = await provider.list_voices()
            selected_voice = next((v for v in voices if v.id == voice_id or v.name == voice_id), None)
            if not selected_voice:
                console.print(f"[yellow]Voice '{voice_id}' not found. Falling back to auto-selection.[/yellow]")
        
        if not selected_voice:
            voices = await provider.list_voices()
            
            if languages:
                filtered_voices = [v for v in voices if any(l.lower() in v.language.lower() for l in languages)]
                if not filtered_voices:
                    console.print(f"[bold red]No voices found matching languages: {', '.join(languages)}[/bold red]")
                    return
                voices = filtered_voices
            
            if not languages:
                selected_voice = next((v for v in voices if "en-US" in v.language), voices[0])
            else:
                selected_voice = voices[0]
        
        console.print(f"Using voice: [cyan]{selected_voice.name}[/cyan] ({selected_voice.id})")

        # 3. Generate Samples
        gen_config = get_generation_config()
        gen_config.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        from wakegen.utils.caching import GenerationCache
        cache = GenerationCache()
        
        cache_hits = 0
        for i in track(range(count), description="Generating samples..."):
            filename = f"{text.replace(' ', '_').lower()}_{i+1}.{gen_config.audio_format}"
            file_path = os.path.join(output_dir, filename)
            
            cached_path = cache.get(text, selected_voice.id, p_type.value)
            if cached_path:
                import shutil
                shutil.copy2(cached_path, file_path)
                cache_hits += 1
            else:
                await provider.generate(text, selected_voice.id, file_path)
                cache.put(text, selected_voice.id, p_type.value, file_path, copy=True)
            
            if gen_config.sample_rate:
                resample_audio(file_path, gen_config.sample_rate)
        
        cache.save_metadata()
        
        if cache_hits > 0:
            console.print(f"[bold green]Successfully generated {count} samples![/bold green] [dim]({cache_hits} from cache)[/dim]")
        else:
            console.print(f"[bold green]Successfully generated {count} samples![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
