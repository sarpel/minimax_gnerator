"""
ChatTTS Provider

ChatTTS is a text-to-speech model optimized for dialogue and conversational
speech. It produces natural-sounding speech with good prosody and rhythm
for conversational contexts.

Features:
- Optimized for conversational/dialogue speech
- Natural prosody and rhythm
- Multiple speaker styles
- Good for wake words with conversational feel
- Supports Chinese and English

Requirements:
- pip install chattts
- GPU recommended for better performance
- Works on CPU but slower

Reference: https://github.com/2noise/ChatTTS
"""

from __future__ import annotations

import asyncio

try:
    import ChatTTS
except ImportError:
    ChatTTS = None
from pathlib import Path
from typing import Any

import numpy as np

from wakegen.core.exceptions import ProviderError
from wakegen.core.types import Gender, ProviderType
from wakegen.models.audio import Voice
from wakegen.models.config import ProviderConfig
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import (
    register_provider,  # CRITICAL: Required for provider discovery
)


class ChatTTSProvider(BaseProvider):
    """
    ChatTTS Provider - Conversational speech synthesis.

    ChatTTS is optimized for generating natural-sounding conversational
    speech. It's particularly good for wake words that need to sound
    like they're part of a natural conversation.

    The model supports:
    - Random speaker generation for diversity
    - Speaker embedding saving/loading for consistency
    - Control over speaking speed and intonation
    """

    # Default speaker IDs (these are seed values for reproducible speakers)
    PRESET_SPEAKERS = {
        "conversational_1": 1234,
        "conversational_2": 5678,
        "conversational_3": 9012,
        "friendly_1": 3456,
        "friendly_2": 7890,
        "professional_1": 2345,
        "professional_2": 6789,
        "casual_1": 1357,
        "casual_2": 2468,
    }

    def __init__(
        self,
        config: ProviderConfig | None = None,
        use_gpu: bool = True,
        compile_model: bool = False,
    ) -> None:
        """
        Initialize the ChatTTS provider.

        Args:
            config: Provider configuration. If None, uses default ProviderConfig.
            use_gpu: Whether to use GPU for inference.
            compile_model: Whether to compile the model with torch.compile
                          (faster but longer startup time).
        """
        super().__init__(config or ProviderConfig())
        self._model = None
        self._use_gpu = use_gpu
        self._compile_model = compile_model
        self._sample_rate = 24000  # ChatTTS outputs 24kHz audio
        self._speaker_cache: dict[str, Any] = {}

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.CHATTTS

    @property
    def requires_api_key(self) -> bool:
        """ChatTTS does not require an API key."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """ChatTTS does not support streaming in this implementation."""
        return False

    async def initialize(self) -> None:
        """Initialize the ChatTTS model."""
        if self._model is not None:
            return

        try:
            import ChatTTS
            import torch

            # Determine device
            if self._use_gpu and torch.cuda.is_available():
                device = "cuda"
            elif (
                self._use_gpu
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

            # Initialize ChatTTS
            chat = ChatTTS.Chat()
            await asyncio.to_thread(
                chat.load,
                compile=self._compile_model,
            )

            self._model = chat
            self._device = device

        except ImportError as e:
            raise ProviderError(
                f"ChatTTS is not installed. Install with: pip install chattts\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise ProviderError(f"Failed to initialize ChatTTS: {e}") from e

    async def generate(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        **kwargs: Any,
    ) -> None:
        """
        Generate speech using ChatTTS.

        Args:
            text: Text to synthesize.
            voice_id: Speaker preset name or seed number.
            output_path: Path to save the audio file.
            **kwargs: Additional parameters:
                - temperature: Sampling temperature (default: 0.3)
                - top_p: Top-p sampling (default: 0.7)
                - top_k: Top-k sampling (default: 20)
                - speed: Speaking speed multiplier (default: 1.0)
                - oral: Oral sound randomness 0-9 (default: 0)
                - laugh: Laughter randomness 0-9 (default: 0)
                - break_pattern: Break pattern randomness 0-9 (default: 0)

        Raises:
            ProviderError: If generation fails.
        """
        await self.initialize()

        try:
            import torch
            from scipy.io.wavfile import write as write_wav

            # Get generation parameters
            temperature = kwargs.get("temperature", 0.3)
            top_p = kwargs.get("top_p", 0.7)
            top_k = kwargs.get("top_k", 20)
            speed = kwargs.get("speed", 1.0)

            # Get control parameters for expressiveness
            oral = kwargs.get("oral", 0)
            laugh = kwargs.get("laugh", 0)
            break_pattern = kwargs.get("break_pattern", 0)

            # Get or create speaker embedding
            speaker = await self._get_speaker_embedding(voice_id)

            # Set up parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker,
                temperature=temperature,
                top_P=top_p,
                top_K=top_k,
            )

            # Add control tokens if specified
            control_text = text
            if oral > 0 or laugh > 0 or break_pattern > 0:
                control_text = (
                    f"[oral_{oral}][laugh_{laugh}][break_{break_pattern}]{text}"
                )

            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt=f"[speed_{int(speed * 5)}]",  # Speed 1-10
            )

            # Generate audio
            assert self._model is not None
            wavs = await asyncio.to_thread(
                self._model.infer,
                [control_text],
                params_infer_code=params_infer_code,
                params_refine_text=params_refine_text,
                use_decoder=True,
            )

            # Get audio and convert to int16
            audio_array = wavs[0]
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            # Normalize and convert to int16
            audio_array = audio_array.flatten()
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Save audio
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            write_wav(str(output_file), self._sample_rate, audio_int16)

        except Exception as e:
            raise ProviderError(f"ChatTTS generation failed: {e}") from e

    async def cleanup(self) -> None:
        """
        Clean up resources.
        Unloads the ChatTTS model to free up memory/VRAM.
        """
        if self._model:
            del self._model
            self._model = None

        # Clear speaker cache
        self._speaker_cache.clear()

        # Force garbage collection
        import gc
        gc.collect()

        # If using GPU, empty cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        await super().cleanup()

    async def _get_speaker_embedding(self, voice_id: str) -> Any:
        """
        Get or create a speaker embedding.

        Args:
            voice_id: Speaker preset name or seed number.

        Returns:
            Speaker embedding for ChatTTS.
        """
        # Check cache first
        if voice_id in self._speaker_cache:
            return self._speaker_cache[voice_id]

        # Get seed from preset or parse as integer
        if voice_id in self.PRESET_SPEAKERS:
            seed = self.PRESET_SPEAKERS[voice_id]
        else:
            try:
                seed = int(voice_id)
            except ValueError:
                # Use hash of voice_id as seed for reproducibility
                seed = hash(voice_id) % (2**31)

        # Generate speaker embedding
        import torch

        torch.manual_seed(seed)

        assert self._model is not None
        # Note: ChatTTS model loading uses torch.load internally which can be unsafe
        # with untrusted files. We are using the official model here.
        speaker = await asyncio.to_thread(
            self._model.sample_random_speaker,
        )

        # Cache the speaker
        self._speaker_cache[voice_id] = speaker

        return speaker

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available speaker presets.

        Args:
            language: Filter by language (not used, ChatTTS supports all).

        Returns:
            List of available voices.
        """
        voices = []

        for preset_name, seed in self.PRESET_SPEAKERS.items():
            # Parse preset name for characteristics
            style = preset_name.split("_")[0]

            voices.append(
                Voice(
                    id=preset_name,
                    name=f"ChatTTS {style.title()} Voice",
                    gender=Gender.NEUTRAL,  # ChatTTS generates varied voices
                    language="multi",  # Supports multiple languages
                    provider=ProviderType.CHATTTS,
                    supports_cloning=False,
                )
            )

        # Add some numeric seed options
        for i in range(5):
            seed = i * 1111
            voices.append(
                Voice(
                    id=str(seed),
                    name=f"ChatTTS Random Voice {i + 1}",
                    gender=Gender.NEUTRAL,
                    language="multi",
                    provider=ProviderType.CHATTTS,
                    supports_cloning=False,
                )
            )

        return voices

    async def check_availability(self) -> bool:
        """Check if ChatTTS is available."""
        try:
            import ChatTTS

            return True
        except ImportError:
            return False

    async def validate_config(self) -> None:
        """
        Validate the provider configuration.

        ChatTTS doesn't require API keys or special configuration,
        so this just verifies the ChatTTS library is available.

        Raises:
            ProviderError: If ChatTTS library is not installed.
        """
        if not await self.check_availability():
            raise ProviderError(
                "ChatTTS is not installed. Install with: pip install chattts"
            )

    def save_speaker_embedding(self, voice_id: str, path: str) -> None:
        """
        Save a speaker embedding to file for later use.

        Args:
            voice_id: The voice ID to save.
            path: Path to save the embedding.
        """
        if voice_id not in self._speaker_cache:
            raise ProviderError(
                f"Speaker {voice_id} not in cache. Generate audio first."
            )

        import torch

        torch.save(self._speaker_cache[voice_id], path)

    def load_speaker_embedding(self, voice_id: str, path: str) -> None:
        """
        Load a speaker embedding from file.

        Args:
            voice_id: The voice ID to assign.
            path: Path to load the embedding from.
        """
        import torch

        self._speaker_cache[voice_id] = torch.load(path)


# =============================================================================
# REGISTER PROVIDER
# =============================================================================
# This registration is CRITICAL - without it, the system won't discover this provider!
register_provider(ProviderType.CHATTTS, ChatTTSProvider)
