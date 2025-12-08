"""
Provider Commands Module

This module contains CLI commands for listing TTS providers and voices.
These commands help users understand what providers are available and
what voices each provider supports.

Commands:
- list-providers: Show all TTS providers and their availability
- list-voices: Show available voices for a provider
"""

import click
import asyncio
from rich.console import Console
from rich.table import Table

from wakegen.providers.registry import (
    get_provider,
    list_available_providers,
    discover_available_providers,
)
from wakegen.core.types import ProviderType
from wakegen.config.settings import get_provider_config


# =============================================================================
# SHARED CONSOLE
# =============================================================================
# We use Rich's Console for beautiful, colorful output.
# This console instance is shared across all provider commands.

console = Console()


# =============================================================================
# LIST-PROVIDERS COMMAND
# =============================================================================


def register_provider_commands(cli_group: click.Group) -> None:
    """
    Register provider-related commands with the main CLI group.
    
    Args:
        cli_group: The main Click group to add commands to.
    """
    
    @cli_group.command(name="list-providers")
    @click.option("--available-only", "-a", is_flag=True, help="Show only providers that are ready to use")
    @click.option("--verbose", "-v", is_flag=True, help="Show detailed information including install hints")
    def list_providers(available_only: bool, verbose: bool) -> None:
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


# =============================================================================
# LIST-VOICES COMMAND
# =============================================================================


    @cli_group.command(name="list-voices")
    @click.option("--provider", default="all", help="Provider to list voices for (default: all)")
    @click.option("--language", "-l", multiple=True, help="Filter voices by language code (e.g., tr-TR)")
    def list_voices(provider: str, language: tuple) -> None:
        """
        Lists available voices for the specified provider(s).
        
        You can see which voices are available for generation.
        Some providers have hundreds of voices!
        
        Examples:
            wakegen list-voices                 # List all voices from all providers
            wakegen list-voices --provider edge_tts  # List only Edge TTS voices
            wakegen list-voices --language tr-TR     # List only Turkish voices
        """
        asyncio.run(_run_list_voices(provider, language))


async def _run_list_voices(provider_name: str, languages: tuple) -> None:
    """
    Async implementation of list-voices.
    
    Args:
        provider_name: Name of provider to list voices for, or "all".
        languages: Tuple of language codes to filter by.
    """
    # 1. Get providers to check
    providers_to_check = []
    if provider_name.lower() == "all":
        providers_to_check = [p.name for p in list_available_providers()]
    else:
        providers_to_check = [provider_name]
    
    provider_config = get_provider_config()
    total_voices = 0
    
    for p_name in providers_to_check:
        try:
            # Get the provider instance
            p_type = ProviderType(p_name.lower())
            provider = get_provider(p_type, provider_config)
            
            # Fetch voices
            console.print(f"[dim]Fetching voices for {p_name}...[/dim]")
            voices = await provider.list_voices()
            
            # Filter by language if requested
            if languages:
                voices = [v for v in voices if any(l.lower() in v.language.lower() for l in languages)]
            
            if not voices:
                if languages:
                    console.print(f"[yellow]No voices found for {p_name} matching languages: {', '.join(languages)}[/yellow]")
                else:
                    console.print(f"[yellow]No voices found for {p_name}[/yellow]")
                continue
            
            total_voices += len(voices)
            
            # Display table
            table = Table(title=f"Voices for {p_name} ({len(voices)})", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Language")
            table.add_column("Gender")
            
            # Show top 50 to avoid spamming
            for v in voices[:50]:
                table.add_row(v.id, v.name, v.language, v.gender.value)
            
            console.print(table)
            if len(voices) > 50:
                console.print(f"[dim]... and {len(voices) - 50} more[/dim]")
            console.print("")
            
        except Exception as e:
            console.print(f"[red]Error listing voices for {p_name}: {e}[/red]")
