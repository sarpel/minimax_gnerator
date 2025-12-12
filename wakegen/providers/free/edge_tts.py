"""
Edge TTS Provider

Microsoft Edge TTS is a free, cloud-based text-to-speech service that provides
high-quality voices in many languages. No API key is required.

Features:
- Free to use (no API key required)
- High-quality neural voices
- Many languages supported including Turkish
- Fast generation (cloud-based)

Installation:
- Already included in pyproject.toml (pip install edge-tts)

Reference: https://github.com/rany2/edge-tts
"""

from __future__ import annotations

from wakegen.core.exceptions import ProviderError
from wakegen.core.types import Gender, ProviderType
from wakegen.models.audio import Voice
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider

# We implement the 'BaseProvider' class to create our Edge TTS provider.
# This class handles the communication with the Microsoft Edge TTS service.
# Note: We use lazy imports for edge_tts inside methods for consistency
# with other providers, even though it's a core dependency.


class EdgeTTSProvider(BaseProvider):
    """
    Provider implementation for Microsoft Edge TTS (Free).

    Edge TTS is a free, cloud-based TTS service from Microsoft.
    It provides high-quality neural voices without requiring an API key.
    """

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type identifier."""
        return ProviderType.EDGE_TTS

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Edge TTS.

        Args:
            text: The text to synthesize.
            voice_id: The voice ID (e.g., "tr-TR-EmelNeural").
            output_path: Path to save the generated audio file.

        Raises:
            ProviderError: If generation fails.
        """
        try:
            # =====================================================================
            # INPUT VALIDATION
            # =====================================================================
            # Validate inputs to provide clear error messages before calling API

            # Check voice_id is not empty (causes "No audio received" error)
            if not voice_id or voice_id.strip() == "":
                raise ProviderError(
                    "Edge TTS voice_id cannot be empty. "
                    "Please provide a valid voice ID like 'tr-TR-EmelNeural' or 'tr-TR-AhmetNeural'."
                )

            # Check text is not empty
            if not text or text.strip() == "":
                raise ProviderError("Edge TTS text cannot be empty.")

            # Lazy import for consistency with other providers
            import edge_tts

            # Create the Communicate object with text and voice
            communicate = edge_tts.Communicate(text, voice_id)

            # Save the audio to the specified path
            await communicate.save(output_path)

        except ImportError as e:
            raise ProviderError(
                f"Edge TTS is not installed. Install with: pip install edge-tts\n"
                f"Original error: {e}"
            ) from e
        except ProviderError:
            # Re-raise our validation errors as-is
            raise
        except Exception as e:
            # Provide more helpful error messages for common issues
            error_msg = str(e)
            if "No audio was received" in error_msg:
                raise ProviderError(
                    f"Edge TTS generation failed: No audio was received. "
                    f"This usually means the voice_id '{voice_id}' is invalid. "
                    f"Try using a valid voice like 'tr-TR-EmelNeural' or 'tr-TR-AhmetNeural'."
                ) from e
            raise ProviderError(f"Edge TTS generation failed: {error_msg}") from e

    async def list_voices(self) -> list[Voice]:
        """
        Lists available voices from Edge TTS.

        Returns:
            List of Voice objects representing available voices.
        """
        try:
            # Lazy import for consistency with other providers
            import edge_tts

            # Get all available voices
            voices = await edge_tts.list_voices()

            # Convert them to our internal 'Voice' model
            voice_list = []
            for v in voices:
                # Determine gender (Edge TTS returns "Male" or "Female")
                gender = Gender.MALE if v["Gender"] == "Male" else Gender.FEMALE

                voice_list.append(
                    Voice(
                        id=v["ShortName"],
                        # Fallback to DisplayName or ShortName if FriendlyName missing
                        name=str(
                            v.get("FriendlyName", v.get("DisplayName", v["ShortName"]))
                        ),
                        gender=gender,
                        language=v["Locale"],
                        provider=self.provider_type,
                        supports_cloning=False,
                    )
                )
            return voice_list

        except ImportError as e:
            raise ProviderError(
                f"Edge TTS is not installed. Install with: pip install edge-tts\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise ProviderError(f"Failed to list Edge TTS voices: {e!s}") from e

    async def validate_config(self) -> None:
        """
        Edge TTS doesn't require API keys, so validation is always successful.
        We just check that the library is installed.
        """
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ProviderError(
                "Edge TTS is not installed. Install with: pip install edge-tts"
            )


# Register this provider so the factory knows about it
register_provider(ProviderType.EDGE_TTS, EdgeTTSProvider)
