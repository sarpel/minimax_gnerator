import click
import asyncio
import os
from typing import Optional
from rich.console import Console
from rich.progress import track
from wakegen.config.settings import get_generation_config, get_provider_config
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
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

@cli.command()
@click.option("--text", help="The wake word text to generate (e.g., 'Hey Katya')")
@click.option("--count", default=1, help="Number of samples to generate")
@click.option("--preset", help="Name of a configuration preset to use (e.g., 'quick_test')")
@click.option("--output-dir", help="Directory to save the output")
@click.option("--interactive", is_flag=True, help="Run in interactive wizard mode")
def generate(text: Optional[str], count: int, preset: Optional[str], output_dir: Optional[str], interactive: bool):
    """
    Generates audio samples for a given wake word.
    
    You can provide arguments directly or use --interactive for a guided wizard.
    """
    # If interactive flag is set OR no text is provided, run the wizard
    if interactive or not text:
        asyncio.run(run_interactive_generation())
    else:
        # Run the async function in the event loop
        asyncio.run(run_generation(text, count, preset, output_dir))

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
        output_dir=config["output_dir"]
        # Note: Provider selection would need to be passed to run_generation
        # For now, run_generation defaults to Edge TTS, but we'll update it below
    )

async def run_generation(text: str, count: int, preset: Optional[str], output_dir: Optional[str]):
    """
    The main logic for the generation command.
    """
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