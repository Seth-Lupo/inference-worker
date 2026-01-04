"""
Triton Inference Server client with async gRPC streaming support.

Provides connection management and inference methods for decoupled models.

Features:
- Async gRPC connection management
- Streaming inference for decoupled models (like CosyVoice2)
- Cancel event support for barge-in
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Dict, List
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
    """Triton client configuration."""
    host: str = "localhost"
    port: int = 8001
    timeout: float = 30.0

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class InferenceResult:
    """Result from Triton inference."""
    outputs: Dict[str, np.ndarray]
    model_name: str


class TritonClient:
    """
    Async Triton gRPC client with streaming support for decoupled models.
    """

    def __init__(self, config: Optional[TritonConfig] = None):
        if not TRITON_AVAILABLE:
            raise RuntimeError("tritonclient not installed")

        self.config = config or TritonConfig()
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to Triton server."""
        async with self._lock:
            if self._connected:
                return

            self._client = grpcclient.InferenceServerClient(
                url=self.config.url,
                verbose=False,
            )

            if await self._client.is_server_live():
                self._connected = True
                logger.info(f"Connected to Triton at {self.config.url}")
            else:
                raise ConnectionError("Triton server not live")

    async def disconnect(self) -> None:
        """Disconnect from Triton server."""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None
                self._connected = False

    async def is_model_ready(self, model_name: str) -> bool:
        """Check if model is ready."""
        if not self._connected:
            return False
        try:
            return await self._client.is_model_ready(model_name)
        except Exception:
            return False

    def _build_inputs(self, inputs: Dict[str, np.ndarray]) -> List:
        """Build Triton input tensors."""
        triton_inputs = []
        for name, data in inputs.items():
            inp = grpcclient.InferInput(
                name,
                list(data.shape),
                np_to_triton_dtype(data.dtype),
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)
        return triton_inputs

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
    ) -> InferenceResult:
        """Non-streaming inference."""
        if not self._connected:
            await self.connect()

        triton_inputs = self._build_inputs(inputs)
        triton_outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]

        result = await self._client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )

        return InferenceResult(
            outputs={n: result.as_numpy(n) for n in output_names},
            model_name=model_name,
        )

    async def infer_stream(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[InferenceResult]:
        """
        Streaming inference for decoupled models.

        Uses the async client's stream_infer() for proper decoupled model support.
        """
        if not self._connected:
            await self.connect()

        logger.info(f"Triton: Starting stream inference for model={model_name}")
        logger.debug(f"Triton: Inputs={list(inputs.keys())}, outputs={output_names}")
        stream_start = time.perf_counter()

        triton_inputs = self._build_inputs(inputs)
        triton_outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]

        result_count = 0

        # Build async request iterator for stream_infer
        async def request_iterator():
            yield {
                "model_name": model_name,
                "inputs": triton_inputs,
                "outputs": triton_outputs,
            }

        try:
            # Use async stream_infer which returns an async generator
            # Response is a tuple of (result, error)
            response_iterator = self._client.stream_infer(
                inputs_iterator=request_iterator(),
            )

            async for response in response_iterator:
                result, error = response

                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    logger.info(f"Triton: Stream cancelled by caller after {result_count} results")
                    response_iterator.cancel()
                    break

                # Check for errors
                if error is not None:
                    raise RuntimeError(f"Stream error: {error}")

                if result is None:
                    continue

                # Extract outputs
                outputs = {}
                for name in output_names:
                    try:
                        arr = result.as_numpy(name)
                        outputs[name] = arr
                        logger.debug(f"Triton: Output '{name}' shape={arr.shape}, dtype={arr.dtype}")
                    except Exception as e:
                        logger.debug(f"Triton: Failed to get output '{name}': {e}")

                if outputs:
                    result_count += 1
                    logger.debug(f"Triton: Yielding result #{result_count}")
                    yield InferenceResult(outputs=outputs, model_name=model_name)

            elapsed = (time.perf_counter() - stream_start) * 1000
            logger.info(f"Triton: Stream complete - {result_count} results in {elapsed:.0f}ms")

        except Exception as e:
            logger.error(f"Triton: Stream error: {e}")
            raise

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()


# Global client
_global_client: Optional[TritonClient] = None
_global_config: Optional[TritonConfig] = None


def configure_triton(config: TritonConfig) -> None:
    """Configure global Triton client."""
    global _global_config
    _global_config = config


async def get_triton_client() -> TritonClient:
    """Get global Triton client."""
    global _global_client, _global_config

    if _global_client is None:
        _global_client = TritonClient(_global_config or TritonConfig())
        await _global_client.connect()

    return _global_client


async def close_triton_client() -> None:
    """Close global client."""
    global _global_client

    if _global_client:
        await _global_client.disconnect()
        _global_client = None
