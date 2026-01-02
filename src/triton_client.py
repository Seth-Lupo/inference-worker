"""
Triton Inference Server client with async gRPC support.

Provides connection management and inference methods for all models.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Dict, Any, List
import numpy as np

try:
    import tritonclient.grpc.aio as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    grpcclient = None
    np_to_triton_dtype = None

logger = logging.getLogger(__name__)


@dataclass
class TritonConfig:
    """Configuration for Triton client connection."""
    host: str = "localhost"
    port: int = 8001  # gRPC port
    verbose: bool = False
    timeout: float = 30.0
    ssl: bool = False
    ssl_cert: Optional[str] = None

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class InferenceResult:
    """Result from a Triton inference request."""
    outputs: Dict[str, np.ndarray]
    model_name: str
    model_version: str = ""


class TritonClient:
    """
    Async Triton Inference Server client.

    Handles connection management, health checks, and inference requests.
    Uses gRPC for optimal streaming performance.
    """

    def __init__(self, config: Optional[TritonConfig] = None):
        """
        Initialize Triton client.

        Args:
            config: Connection configuration (uses defaults if None)
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "tritonclient not installed. Install with: "
                "pip install tritonclient[grpc]"
            )

        self.config = config or TritonConfig()
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._connected = False
        self._lock = asyncio.Lock()

        logger.debug(f"TritonClient initialized for {self.config.url}")

    async def connect(self) -> None:
        """Establish connection to Triton server."""
        async with self._lock:
            if self._connected:
                return

            try:
                self._client = grpcclient.InferenceServerClient(
                    url=self.config.url,
                    verbose=self.config.verbose,
                )

                # Verify connection
                if await self._client.is_server_live():
                    self._connected = True
                    logger.info(f"Connected to Triton at {self.config.url}")
                else:
                    raise ConnectionError("Triton server is not live")

            except Exception as e:
                logger.error(f"Failed to connect to Triton: {e}")
                raise

    async def disconnect(self) -> None:
        """Close connection to Triton server."""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None
                self._connected = False
                logger.info("Disconnected from Triton")

    async def is_healthy(self) -> bool:
        """Check if Triton server is healthy."""
        if not self._connected or not self._client:
            return False
        try:
            return await self._client.is_server_ready()
        except Exception:
            return False

    async def is_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready for inference."""
        if not self._connected or not self._client:
            return False
        try:
            return await self._client.is_model_ready(model_name)
        except Exception:
            return False

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        model_version: str = "",
        timeout: Optional[float] = None,
    ) -> InferenceResult:
        """
        Run inference on a model (non-streaming).

        Args:
            model_name: Name of the model
            inputs: Dict mapping input names to numpy arrays
            output_names: List of output tensor names to retrieve
            model_version: Specific model version (empty for latest)
            timeout: Request timeout in seconds

        Returns:
            InferenceResult with output tensors
        """
        if not self._connected:
            await self.connect()

        # Build input tensors
        triton_inputs = []
        for name, data in inputs.items():
            inp = grpcclient.InferInput(
                name,
                list(data.shape),
                np_to_triton_dtype(data.dtype),
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # Build output requests
        triton_outputs = [
            grpcclient.InferRequestedOutput(name)
            for name in output_names
        ]

        # Run inference
        result = await self._client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            model_version=model_version,
            client_timeout=timeout or self.config.timeout,
        )

        # Extract outputs
        outputs = {
            name: result.as_numpy(name)
            for name in output_names
        }

        return InferenceResult(
            outputs=outputs,
            model_name=model_name,
            model_version=model_version,
        )

    async def infer_stream(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        model_version: str = "",
        timeout: Optional[float] = None,
    ) -> AsyncIterator[InferenceResult]:
        """
        Run streaming inference on a decoupled model.

        For models configured with decoupled=True, this yields
        multiple results as they become available.

        Args:
            model_name: Name of the model
            inputs: Dict mapping input names to numpy arrays
            output_names: List of output tensor names to retrieve
            model_version: Specific model version (empty for latest)
            timeout: Request timeout in seconds

        Yields:
            InferenceResult for each streaming response
        """
        if not self._connected:
            await self.connect()

        # Build input tensors
        triton_inputs = []
        for name, data in inputs.items():
            inp = grpcclient.InferInput(
                name,
                list(data.shape),
                np_to_triton_dtype(data.dtype),
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # Build output requests
        triton_outputs = [
            grpcclient.InferRequestedOutput(name)
            for name in output_names
        ]

        # Use streaming interface
        async for result in self._client.stream_infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            model_version=model_version,
        ):
            # Handle potential errors in stream
            if result is None:
                continue

            # Check for error response
            if hasattr(result, 'get_error') and result.get_error():
                raise RuntimeError(f"Triton streaming error: {result.get_error()}")

            # Extract outputs
            outputs = {}
            for name in output_names:
                try:
                    outputs[name] = result.as_numpy(name)
                except Exception:
                    # Output might not be present in all responses
                    pass

            if outputs:
                yield InferenceResult(
                    outputs=outputs,
                    model_name=model_name,
                    model_version=model_version,
                )

    async def __aenter__(self) -> "TritonClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()


# Global client instance (lazy initialization)
_global_client: Optional[TritonClient] = None
_global_config: Optional[TritonConfig] = None


def configure_triton(config: TritonConfig) -> None:
    """Configure the global Triton client."""
    global _global_config
    _global_config = config


async def get_triton_client() -> TritonClient:
    """Get or create the global Triton client instance."""
    global _global_client, _global_config

    if _global_client is None:
        config = _global_config or TritonConfig()
        _global_client = TritonClient(config)
        await _global_client.connect()

    return _global_client


async def close_triton_client() -> None:
    """Close the global Triton client."""
    global _global_client

    if _global_client is not None:
        await _global_client.disconnect()
        _global_client = None
