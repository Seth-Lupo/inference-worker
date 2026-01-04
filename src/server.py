"""
WebSocket server for voice agent connections.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict

import websockets
from websockets.server import WebSocketServerProtocol

from .agent_rail import AgentRail, AgentEvent
from .tts_rail import create_tts_rail, TTSBackend

logger = logging.getLogger(__name__)


class VoiceAgentServer:
    """
    WebSocket server that handles concurrent voice agent sessions.

    Protocol:
    - Client sends binary audio data (PCM16, 16kHz, mono)
    - Server sends binary audio data (PCM16, 16kHz, mono)
    - Server sends JSON messages for events (transcript, state, etc.)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 80,
        tts_backend: str = "mock",
        triton_url: str = "localhost:8001",
        tts_model: str = "chatterbox",
        reference_audio_path: str = None,
        reference_text: str = None,
    ):
        self.host = host
        self.port = port
        self._tts_backend = TTSBackend(tts_backend)
        self._triton_url = triton_url
        self._tts_model = tts_model
        self._reference_audio_path = reference_audio_path
        self._reference_text = reference_text
        self._sessions: Dict[str, AgentRail] = {}
        self._server = None
        logger.debug(f"VoiceAgentServer created: {host}:{port}, TTS: {tts_backend}")
        if reference_audio_path:
            logger.info(f"Voice cloning enabled: {reference_audio_path}")

    async def start(self) -> None:
        """Start the WebSocket server."""
        logger.debug("Starting WebSocket server...")
        start_time = time.perf_counter()

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=1024 * 1024,  # 1MB max message size
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Voice Agent Server started on ws://{self.host}:{self.port} ({elapsed_ms:.1f}ms)")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        logger.debug("Stopping WebSocket server...")
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Voice Agent Server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection."""
        session_id = str(uuid.uuid4())[:8]
        remote_addr = websocket.remote_address
        logger.info(f"New connection: session={session_id} from={remote_addr}")

        # Create TTS rail for this session
        logger.debug(f"[{session_id}] Creating TTS rail: {self._tts_backend.value}")
        tts = create_tts_rail(
            backend=self._tts_backend,
            triton_url=self._triton_url,
            model_name=self._tts_model,
            reference_audio_path=self._reference_audio_path,
            reference_text=self._reference_text,
        )

        # Create agent rail for this session
        logger.debug(f"[{session_id}] Creating AgentRail...")
        agent_start = time.perf_counter()
        agent = AgentRail(tts=tts)
        self._sessions[session_id] = agent
        logger.debug(f"[{session_id}] AgentRail created in {(time.perf_counter() - agent_start)*1000:.1f}ms")

        # Send connected message
        await websocket.send(json.dumps({
            "type": "connected",
            "session_id": session_id
        }))
        logger.debug(f"[{session_id}] Sent connected message")

        message_count = 0
        audio_bytes_received = 0

        try:
            async for message in websocket:
                message_count += 1

                if isinstance(message, bytes):
                    # Binary message = audio data
                    audio_bytes_received += len(message)
                    logger.debug(f"[{session_id}] Audio message #{message_count}: {len(message)} bytes (total: {audio_bytes_received})")
                    await self._handle_audio(websocket, agent, message, session_id)
                else:
                    # Text message = JSON control message
                    logger.debug(f"[{session_id}] Control message #{message_count}: {message[:100]}")
                    await self._handle_control(websocket, agent, message, session_id)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{session_id}] Connection closed: code={e.code} reason={e.reason}")
        except Exception as e:
            logger.error(f"[{session_id}] Error: {e}", exc_info=True)
        finally:
            # Cleanup
            logger.debug(f"[{session_id}] Cleaning up session...")
            agent.reset()
            del self._sessions[session_id]
            logger.info(f"[{session_id}] Session ended: {message_count} messages, {audio_bytes_received} audio bytes")

    async def _handle_audio(
        self,
        websocket: WebSocketServerProtocol,
        agent: AgentRail,
        audio_data: bytes,
        session_id: str
    ) -> None:
        """Process incoming audio data."""
        try:
            async for event in agent.process_audio(audio_data):
                await self._send_event(websocket, event, session_id)
        except Exception as e:
            logger.error(f"[{session_id}] Error processing audio: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def _handle_control(
        self,
        websocket: WebSocketServerProtocol,
        agent: AgentRail,
        message: str,
        session_id: str
    ) -> None:
        """Handle control messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            logger.debug(f"[{session_id}] Control message type: {msg_type}")

            if msg_type == "reset":
                logger.info(f"[{session_id}] Reset requested")
                agent.reset()
                await websocket.send(json.dumps({"type": "reset_ack"}))

            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            else:
                logger.warning(f"[{session_id}] Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"[{session_id}] Invalid JSON: {e}")

    async def _send_event(
        self,
        websocket: WebSocketServerProtocol,
        event: AgentEvent,
        session_id: str
    ) -> None:
        """Send an agent event to the client."""
        if event.type == "audio":
            # Send audio as binary
            logger.debug(f"[{session_id}] Sending audio: {len(event.data)} bytes")
            await websocket.send(event.data)
        else:
            # Send other events as JSON
            logger.debug(f"[{session_id}] Sending event: type={event.type} data={event.data}")
            await websocket.send(json.dumps({
                "type": event.type,
                "data": event.data
            }))

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


async def run_server(
    host: str = "0.0.0.0",
    port: int = 80,
    tts_backend: str = "mock",
    triton_url: str = "localhost:8001",
    tts_model: str = "chatterbox",
    reference_audio_path: str = None,
    reference_text: str = None,
) -> None:
    """Run the voice agent server."""
    logger.info(f"Starting voice agent server on {host}:{port}")
    server = VoiceAgentServer(
        host=host,
        port=port,
        tts_backend=tts_backend,
        triton_url=triton_url,
        tts_model=tts_model,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
    )
    await server.start()

    # Keep running until interrupted
    try:
        logger.debug("Server running, waiting for connections...")
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        logger.debug("Server received cancel signal")
    finally:
        await server.stop()
