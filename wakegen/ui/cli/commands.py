import click
import asyncio
import os
from typing import Optional
from rich.console import Console
from rich.progress import track
from wakegen.config.settings import get_generation_config, get_provider_config
from wakegen.providers.registry import get_provider, list_available_providers
from wakegen.core.types import ProviderType
from wakegen.utils.logging import setup_logging
from wakegen.utils.audio import resample_audio
from wakegen.ui.cli.wizard import run_wizard
from wakegen.config.yaml_loader import load_config
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
@click.option("--available", is_flag=True, help="Show only available providers (currently all registered)")
def list_providers(available: bool):
    """
    Lists all supported TTS providers.
    
    Use this to see which providers you can use with the --provider flag.
    """
    # Fetch the list of providers from our registry
    providers = list_available_providers()
    
    console.print("[bold cyan]Available TTS Providers:[/bold cyan]")
    for p in providers:
        # Print each provider name as a bullet point
        console.print(f" - [green]{p.value}[/green]")

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
def generate(text: Optional[str], count: int, preset: Optional[str], output_dir: Optional[str], provider: str, voice: Optional[str], interactive: bool):
    """
    Generates audio samples for a given wake word.
    
    You can provide arguments directly or use --interactive for a guided wizard.
    """
    # If interactive flag is set OR no text is provided, run the wizard
    if interactive or not text:
        asyncio.run(run_interactive_generation())
    else:
        # Run the async function in the event loop
        asyncio.run(run_generation(text, count, preset, output_dir, provider, voice))

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

async def run_generation(text: str, count: int, preset: Optional[str], output_dir: Optional[str], provider_name: str = "edge_tts", voice_id: Optional[str] = None):
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

@cli.group()
def config():
    """
    Manage configuration files.
    """
    pass

@config.command(name="init")
def config_init():
    """
    Generates a template wakegen.yaml configuration file in the current directory.
    """
    # This template matches the structure defined in wakegen/config/yaml_loader.py
    template = """# wakegen.yaml
project:
  name: "hey_katya"
  version: "1.0.0"

generation:
  wake_words:
    - "hey katya"
    - "katya"
  count: 1000
  output_dir: "./output/hey_katya"

providers:
  - type: kokoro
    voices: [af_bella, am_adam]
    weight: 0.4
  - type: piper
    voices: [tr_TR-dfki-medium]
    weight: 0.3
  - type: edge_tts
    voices: [tr-TR-PinarNeural, tr-TR-AhmetNeural]
    weight: 0.3

augmentation:
  enabled: true
  profiles:
    - morning_kitchen
    - car_interior
  augmented_per_original: 3

export:
  format: openwakeword
  split_ratio: [0.8, 0.1, 0.1]
"""
    output_path = "wakegen.yaml"
    
    if os.path.exists(output_path):
        console.print(f"[bold red]Error:[/bold red] '{output_path}' already exists in the current directory.")
        return

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template)
        console.print(f"[bold green]Success:[/bold green] Created '{output_path}'")
        console.print("You can now edit this file to configure your project.")
    except Exception as e:
        console.print(f"[bold red]Error creating file:[/bold red] {str(e)}")

@config.command(name="validate")
@click.argument("path", type=click.Path(exists=True))
def config_validate(path: str):
    """
    Validates a configuration file.
    
    PATH is the path to the YAML configuration file to validate.
    """
    console.print(f"Validating configuration file: [cyan]{path}[/cyan]...")
    
    try:
        # Attempt to load and validate the config
        config = load_config(path)
        
        console.print("[bold green]Configuration is valid![/bold green]")
        
        # Display a brief summary of the loaded configuration
        console.print("\n[bold]Project Summary:[/bold]")
        console.print(f"  Name: [green]{config.project.name}[/green]")
        console.print(f"  Wake Words: [cyan]{', '.join(config.generation.wake_words)}[/cyan]")
        console.print(f"  Output Directory: {config.generation.output_dir}")
        console.print(f"  Providers: {len(config.providers)}")
        console.print(f"  Augmentation: {'[green]Enabled[/green]' if config.augmentation.enabled else '[red]Disabled[/red]'}")
        
    except Exception as e:
        console.print(f"[bold red]Validation Failed:[/bold red]")
        # We print the error message directly, which might contain Pydantic validation details
        console.print(f"{str(e)}")