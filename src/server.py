"""
WebSocket server for voice agent connections.
"""
import asyncio
import json
import logging
import uuid
from typing import Dict

import websockets
from websockets.server import WebSocketServerProtocol

from .agent_rail import AgentRail, AgentEvent

logger = logging.getLogger(__name__)


class VoiceAgentServer:
    """
    WebSocket server that handles concurrent voice agent sessions.

    Protocol:
    - Client sends binary audio data (PCM16, 16kHz, mono)
    - Server sends binary audio data (PCM16, 16kHz, mono)
    - Server sends JSON messages for events (transcript, state, etc.)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 80):
        self.host = host
        self.port = port
        self._sessions: Dict[str, AgentRail] = {}
        self._server = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=1024 * 1024,  # 1MB max message size
        )
        logger.info(f"Voice Agent Server started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Voice Agent Server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection."""
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"New connection: {session_id} from {websocket.remote_address}")

        # Create agent rail for this session
        agent = AgentRail()
        self._sessions[session_id] = agent

        # Send connected message
        await websocket.send(json.dumps({
            "type": "connected",
            "session_id": session_id
        }))

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary message = audio data
                    await self._handle_audio(websocket, agent, message)
                else:
                    # Text message = JSON control message
                    await self._handle_control(websocket, agent, message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {session_id} ({e.code})")
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        finally:
            # Cleanup
            agent.reset()
            del self._sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")

    async def _handle_audio(
        self,
        websocket: WebSocketServerProtocol,
        agent: AgentRail,
        audio_data: bytes
    ) -> None:
        """Process incoming audio data."""
        try:
            async for event in agent.process_audio(audio_data):
                await self._send_event(websocket, event)
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def _handle_control(
        self,
        websocket: WebSocketServerProtocol,
        agent: AgentRail,
        message: str
    ) -> None:
        """Handle control messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "reset":
                agent.reset()
                await websocket.send(json.dumps({"type": "reset_ack"}))

            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")

    async def _send_event(
        self,
        websocket: WebSocketServerProtocol,
        event: AgentEvent
    ) -> None:
        """Send an agent event to the client."""
        if event.type == "audio":
            # Send audio as binary
            await websocket.send(event.data)
        else:
            # Send other events as JSON
            await websocket.send(json.dumps({
                "type": event.type,
                "data": event.data
            }))

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


async def run_server(host: str = "0.0.0.0", port: int = 80) -> None:
    """Run the voice agent server."""
    server = VoiceAgentServer(host, port)
    await server.start()

    # Keep running until interrupted
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
