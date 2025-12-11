import asyncio
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import httpx

from wakegen.core.types import ProviderType
from wakegen.models.config import ProviderConfig
from wakegen.providers.base import BaseProvider
from wakegen.utils.docker_bridge import DockerBridge

logger = logging.getLogger(__name__)


class DockerBaseProvider(BaseProvider):
    """
    Base class for providers that run inside a Docker container (Sidecar pattern).
    Handles lifecycle: Build -> Start -> Health Check -> Proxy Request -> Stop.
    """

    def __init__(
        self,
        config: ProviderConfig,
        image_tag: str,
        container_port: int = 5000,
        host_port: int = 5050,
        dockerfile_dir: Optional[str] = None,
    ):
        super().__init__(config)
        self.image_tag = image_tag
        self.container_name = f"wakegen-sidecar-{self.provider_type.value}"
        self.container_port = container_port
        self.host_port = host_port
        self.dockerfile_dir = dockerfile_dir
        self.base_url = f"http://localhost:{host_port}"
        self._is_ready = False

    async def ensure_running(self) -> bool:
        """
        Ensures the sidecar container is running and healthy.
        Builds the image if missing.
        """
        if self._is_ready:
            return True

        if not DockerBridge.is_docker_available():
            logger.error("Docker is not available. Cannot start sidecar provider.")
            return False

        # 1. Build if we have a source directory
        if self.dockerfile_dir:
            path = Path(self.dockerfile_dir)
            if path.exists():
                # In a real app, we might check if image exists first to skip build
                success = await DockerBridge.build_image(self.image_tag, str(path))
                if not success:
                    return False

        # 2. Start Container
        started = await DockerBridge.start_container(
            self.image_tag, self.container_name, (self.host_port, self.container_port)
        )
        if not started:
            return False

        # 3. Wait for Health Check
        return await self._wait_for_health()

    async def _wait_for_health(self, attempts: int = 10) -> bool:
        """Poll the container's /health endpoint."""
        logger.info(f"Waiting for {self.container_name} to be healthy...")
        async with httpx.AsyncClient() as client:
            for _ in range(attempts):
                try:
                    resp = await client.get(f"{self.base_url}/health", timeout=1.0)
                    if resp.status_code == 200:
                        self._is_ready = True
                        logger.info(f"{self.container_name} is ready!")
                        return True
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1.0)
        
        logger.error(f"Container {self.container_name} timed out.")
        await self.cleanup()
        return False

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Proxies the generation request to the sidecar container.
        """
        if not await self.ensure_running():
            raise RuntimeError(f"Docker provider {self.provider_type} is not available.")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # We expect the sidecar to accept a POST /generate
                # Body: {"text": "...", "voice_id": "..."}
                # Response: Audio binary or JSON with base64
                response = await client.post(
                    f"{self.base_url}/generate",
                    json={"text": text, "voice_id": voice_id},
                )
                response.raise_for_status()

                # Assuming the response is the raw audio file
                with open(output_path, "wb") as f:
                    f.write(response.content)

            except httpx.HTTPError as e:
                logger.error(f"Sidecar generation failed: {e}")
                raise

    async def list_voices(self) -> list[Any]:
        """Proxies voice list request."""
        if not await self.ensure_running():
            return []

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/voices")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.error(f"Failed to list voices from sidecar: {e}")
                return []

    async def cleanup(self) -> None:
        """Stop the container on shutdown."""
        await DockerBridge.stop_container(self.container_name)
        self._is_ready = False
