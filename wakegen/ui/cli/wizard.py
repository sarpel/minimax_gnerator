from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.panel import Panel
from rich.text import Text

from wakegen.config.settings import GenerationConfig
from wakegen.core.types import ProviderType

# Initialize the Rich console for beautiful output
console = Console()

async def run_wizard() -> Dict[str, Any]:
    """
    Runs an interactive wizard to gather configuration from the user.
    
    This function asks the user a series of questions to build a configuration
    dictionary. It's like a setup assistant that guides you step-by-step.
    
    Returns:
        Dict[str, Any]: A dictionary containing the user's choices.
    """
    
    # Display a welcome message in a nice panel
    console.print(Panel.fit(
        "[bold cyan]Welcome to the Wake Word Generator Wizard![/bold cyan]\n"
        "I will guide you through creating your custom dataset.\n"
        "Let's get started!",
        title="WakeGen Wizard"
    ))

    config: Dict[str, Any] = {}

    # --- Step 1: Wake Word ---
    # We need to know what phrase the model should listen for.
    console.print("\n[bold yellow]Step 1: Wake Word Configuration[/bold yellow]")
    console.print("The wake word is the phrase that activates your device (e.g., 'Hey Computer').")
    
    wake_word = Prompt.ask(
        "[green]Enter your wake word(s)[/green]", 
        default="hey computer"
    )
    config["wake_word"] = wake_word

    # --- Step 2: Sample Count ---
    # How many different versions of the wake word do we need?
    # More samples usually mean better training but take longer to generate.
    console.print("\n[bold yellow]Step 2: Dataset Size[/bold yellow]")
    console.print("How many samples do you want to generate?")
    console.print("For a quick test, 10-50 is good. For training, aim for 100+.")
    
    count = IntPrompt.ask(
        "[green]Number of samples[/green]", 
        default=50
    )
    config["count"] = count

    # --- Step 3: Provider Selection ---
    # Which Text-to-Speech engine should we use?
    console.print("\n[bold yellow]Step 3: Voice Provider[/bold yellow]")
    console.print("Select the engine to generate speech.")
    console.print("1. [cyan]Edge TTS[/cyan] (Free, high quality, requires internet)")
    console.print("2. [cyan]Piper TTS[/cyan] (Local, fast, works offline)")
    
    provider_choice = IntPrompt.ask(
        "[green]Select provider[/green]", 
        choices=["1", "2"], 
        default=1
    )
    
    if provider_choice == 1:
        config["provider"] = ProviderType.EDGE_TTS
    else:
        config["provider"] = ProviderType.PIPER

    # --- Step 4: Output Settings ---
    # Where should we save the files?
    console.print("\n[bold yellow]Step 4: Output Settings[/bold yellow]")
    
    output_dir = Prompt.ask(
        "[green]Output directory[/green]", 
        default="./output"
    )
    config["output_dir"] = output_dir

    # --- Step 5: Augmentation ---
    # Do we want to add noise/effects to make the model robust?
    console.print("\n[bold yellow]Step 5: Augmentation[/bold yellow]")
    console.print("Augmentation adds noise and effects to make the model work in real environments.")
    
    use_augmentation = Confirm.ask(
        "[green]Enable augmentation?[/green]", 
        default=True
    )
    config["use_augmentation"] = use_augmentation

    if use_augmentation:
        console.print("  [dim]We'll use the default noise profile (background noise, reverb).[/dim]")

    # --- Summary ---
    console.print("\n[bold green]Configuration Complete![/bold green]")
    console.print(Panel(
        f"Wake Word: [cyan]{config['wake_word']}[/cyan]\n"
        f"Count: [cyan]{config['count']}[/cyan]\n"
        f"Provider: [cyan]{config['provider']}[/cyan]\n"
        f"Output: [cyan]{config['output_dir']}[/cyan]\n"
        f"Augmentation: [cyan]{config['use_augmentation']}[/cyan]",
        title="Summary"
    ))

    start_now = Confirm.ask("\n[bold]Start generation now?[/bold]", default=True)
    
    if start_now:
        return config
    else:
        console.print("[yellow]Exiting wizard without generating.[/yellow]")
        return {}

if __name__ == "__main__":
    # This allows testing the wizard directly
    asyncio.run(run_wizard())