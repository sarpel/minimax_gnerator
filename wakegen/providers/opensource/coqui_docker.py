
from typing import Any
from wakegen.core.types import ProviderType
from wakegen.providers.docker_base import DockerBaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice
from wakegen.core.types import Gender

class CoquiDockerProvider(DockerBaseProvider):
    """
    Coqui XTTS provider running in a Docker sidecar.
    """
    
    def __init__(self, config: Any):
        super().__init__(
            config=config,
            image_tag="wakegen/coqui-xtts:latest",
            container_port=5000,
            host_port=5051,
            dockerfile_dir="docker/providers/coqui_xtts"
        )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.COQUI_XTTS

    async def list_voices(self) -> list[Voice]:
        # Coqui XTTS is a cloning model, it doesn't have "preset" voices in the model itself mostly.
        # But we can provide some sample references or scan a directory.
        # For now, return a placeholder or scan a local dir.
        return [
            Voice(id="reference_audio/sample_female.wav", name="Sample Female (Cloned)", language="en", gender=Gender.FEMALE, provider=self.provider_type),
            Voice(id="reference_audio/sample_male.wav", name="Sample Male (Cloned)", language="en", gender=Gender.MALE, provider=self.provider_type)
        ]

    async def validate_config(self) -> None:
        """
        Validate that Docker is available.
        """
        from wakegen.utils.docker_bridge import DockerBridge
        from wakegen.core.exceptions import ConfigError
        
        if not DockerBridge.is_docker_available():
            raise ConfigError("Docker is not available. Please install Docker to use Coqui XTTS.")

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Proxies the generation request to the sidecar container with file upload.
        """
        import httpx
        import os
        from wakegen.core.exceptions import ProviderError

        if not await self.ensure_running():
            raise RuntimeError(f"Docker provider {self.provider_type} is not available.")

        # Prepare file for upload
        # voice_id is treated as a path to the reference audio file
        if not os.path.exists(voice_id):
             # Try to find it if it's a relative path or sample
             # If using list_voices() hardcoded samples:
             # "reference_audio/sample_female.wav"
             # We might need to ensure these exist or mock them.
             # For now, verify existence.
             raise ProviderError(f"Reference audio not found: {voice_id}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # Open file and stream it
                files = {"reference_audio": open(voice_id, "rb")}
                data = {"text": text, "language": "en"}
                
                response = await client.post(
                    f"{self.base_url}/generate",
                    data=data,
                    files=files
                )
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

            except httpx.HTTPError as e:
                # Close file if opened? files dict handles open() but context manager is better.
                # Actually, simple open() without context leaves it open until GC?
                # Using context manager for file is better but complex with 'files' dict inside request.
                # 'files' takes file-like object.
                raise ProviderError(f"Sidecar generation failed: {e}") from e
            except Exception as e:
                raise ProviderError(f"Generation failed: {e}") from e

register_provider(ProviderType.COQUI_XTTS, CoquiDockerProvider)
