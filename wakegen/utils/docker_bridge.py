import asyncio
import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class DockerBridge:
    """
    A lightweight wrapper around the system's Docker CLI.
    Used to manage 'Sidecar' containers for incompatible providers.
    """

    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker is installed and running."""
        if not shutil.which("docker"):
            return False
        try:
            # check if daemon is responsive
            subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    async def build_image(tag: str, context_path: str) -> bool:
        """
        Build a Docker image from a context directory.
        Returns True if successful.
        """
        logger.info(f"Building Docker image '{tag}' from {context_path}...")
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "-t",
                tag,
                context_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Docker build failed for {tag}:\n{stderr.decode()}")
                return False

            logger.info(f"Docker image '{tag}' built successfully.")
            return True
        except Exception as e:
            logger.error(f"Error building docker image {tag}: {e}")
            return False

    @staticmethod
    async def start_container(
        image_tag: str,
        container_name: str,
        port_mapping: tuple[int, int]
    ) -> bool:
        """
        Start a container in detached mode.
        port_mapping: (host_port, container_port)
        """
        host_port, container_port = port_mapping
        logger.info(f"Starting container '{container_name}' on port {host_port}...")

        # Stop existing if any
        await DockerBridge.stop_container(container_name)

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "-p",
                f"{host_port}:{container_port}",
                image_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to start container {container_name}:\n{stderr.decode()}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            return False

    @staticmethod
    async def stop_container(container_name: str) -> None:
        """Stop and remove a container if it exists."""
        try:
            # We use 'subprocess.run' here for a quick synchronous check/kill if needed,
            # but asyncio is better for the main app flow.
            # Using 'docker rm -f' ensures it's killed and removed.
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except Exception:
            pass
