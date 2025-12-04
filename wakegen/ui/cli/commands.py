import click
import asyncio
import os
from rich.console import Console
from rich.progress import track
from wakegen.config.settings import get_generation_config, get_provider_config
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
from wakegen.utils.logging import setup_logging
from wakegen.utils.audio import resample_audio
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
    """
    setup_logging()

@cli.command()
@click.option("--text", required=True, help="The wake word text to generate (e.g., 'Hey Katya')")
@click.option("--count", default=1, help="Number of samples to generate")
@click.option("--preset", help="Name of a configuration preset to use (e.g., 'quick_test')")
@click.option("--output-dir", help="Directory to save the output")
def generate(text: str, count: int, preset: str, output_dir: str):
    """
    Generates audio samples for a given wake word.
    """
    # Run the async function in the event loop
    asyncio.run(run_generation(text, count, preset, output_dir))

async def run_generation(text: str, count: int, preset: str, output_dir: str):
    """
    The main logic for the generation command.
    """
    try:
        # 1. Load Configuration
        gen_config = get_generation_config(preset)
        provider_config = get_provider_config()

        # Override output directory if provided via CLI
        if output_dir:
            gen_config.output_dir = output_dir

        console.print(f"[bold green]Starting generation for:[/bold green] '{text}'")
        console.print(f"Output directory: {gen_config.output_dir}")

        # 2. Get the Provider (Edge TTS for now)
        # In the future, we could make this configurable via CLI too
        provider_type = ProviderType.EDGE_TTS
        provider = get_provider(provider_type, provider_config)

        # 3. Get a voice to use
        # For Phase 1A, we'll just pick the first English male voice we find
        console.print("Fetching available voices...")
        voices = await provider.list_voices()
        
        # Simple filter for an English voice
        selected_voice = next((v for v in voices if v.language.startswith("en-") and "Neural" in v.id), None)
        
        if not selected_voice:
            console.print("[bold red]No suitable voice found![/bold red]")
            return

        console.print(f"Using voice: [cyan]{selected_voice.name}[/cyan] ({selected_voice.id})")

        # 4. Generate Samples
        os.makedirs(gen_config.output_dir, exist_ok=True)
        
        for i in track(range(count), description="Generating samples..."):
            filename = f"{text.replace(' ', '_').lower()}_{i+1}.{gen_config.audio_format}"
            file_path = os.path.join(gen_config.output_dir, filename)
            
            # Generate the audio
            await provider.generate(text, selected_voice.id, file_path)
            
            # Resample if needed (Edge TTS usually outputs 24kHz, we might want 16kHz)
            if gen_config.sample_rate:
                resample_audio(file_path, gen_config.sample_rate)

        console.print(f"[bold green]Successfully generated {count} samples![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        # In a real app, we might want to print the full traceback here
        # console.print_exception()