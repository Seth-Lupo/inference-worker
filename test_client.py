#!/usr/bin/env python3
"""
Test client for the Voice Agent.

This client captures audio from your microphone and sends it to the server,
then plays back the audio response (beep).

Requirements:
    pip install websockets pyaudio numpy

Usage:
    python test_client.py                    # Connect to localhost:80
    python test_client.py --port 8080        # Connect to localhost:8080
    python test_client.py --host 192.168.1.1 # Connect to remote host
"""
import argparse
import asyncio
import json
import logging
import struct
import sys
import threading
from queue import Queue

try:
    import pyaudio
except ImportError:
    print("PyAudio not installed. Install with: pip install pyaudio")
    print("On Mac: brew install portaudio && pip install pyaudio")
    sys.exit(1)

import websockets
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512  # ~32ms at 16kHz
FORMAT = pyaudio.paInt16


class AudioClient:
    def __init__(self, host: str = "localhost", port: int = 80):
        self.host = host
        self.port = port
        self.ws = None
        self.running = False

        # Audio
        self.pyaudio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.output_queue: Queue = Queue()

    async def connect(self):
        """Connect to the voice agent server."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to {uri}...")

        self.ws = await websockets.connect(uri)
        logger.info("Connected!")

        # Wait for connected message
        msg = await self.ws.recv()
        data = json.loads(msg)
        if data.get("type") == "connected":
            logger.info(f"Session ID: {data.get('session_id')}")

        self.running = True

    def start_audio(self):
        """Start audio input/output streams."""
        # Input stream (microphone)
        self.input_stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Output stream (speakers)
        self.output_stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info("Audio streams started")

    def stop_audio(self):
        """Stop audio streams."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.pyaudio.terminate()

    async def send_audio(self):
        """Send microphone audio to the server."""
        logger.info("Listening... Speak into your microphone!")

        while self.running:
            try:
                # Read audio from microphone
                audio_data = self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Send to server
                await self.ws.send(audio_data)

                # Small delay to prevent flooding
                await asyncio.sleep(0.001)

            except Exception as e:
                if self.running:
                    logger.error(f"Error sending audio: {e}")
                break

    async def receive_messages(self):
        """Receive messages from the server."""
        while self.running:
            try:
                msg = await self.ws.recv()

                if isinstance(msg, bytes):
                    # Audio data - play it
                    self.output_stream.write(msg)

                else:
                    # JSON message
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "state":
                        state = data.get("data", {})
                        if state.get("user_speaking"):
                            logger.info("🎤 User speaking...")
                        elif state.get("processing"):
                            logger.info("⚙️ Processing...")
                        elif state.get("agent_speaking"):
                            logger.info("🔊 Agent speaking...")
                        elif "user_speaking" in state and not state["user_speaking"]:
                            logger.info("🛑 User stopped speaking")
                        elif "agent_speaking" in state and not state["agent_speaking"]:
                            logger.info("✅ Agent finished speaking")

                    elif msg_type == "transcript":
                        text = data.get("data", {}).get("text", "")
                        logger.info(f"📝 Transcript: {text}")

                    elif msg_type == "agent_text":
                        text = data.get("data", {}).get("text", "")
                        is_complete = data.get("data", {}).get("is_complete", False)
                        if is_complete:
                            logger.info(f"🤖 Agent: {text}")

                    elif msg_type == "error":
                        logger.error(f"❌ Error: {data.get('message')}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed")
                break
            except Exception as e:
                if self.running:
                    logger.error(f"Error receiving: {e}")
                break

    async def run(self):
        """Run the client."""
        try:
            await self.connect()
            self.start_audio()

            # Run send and receive concurrently
            await asyncio.gather(
                self.send_audio(),
                self.receive_messages(),
            )

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.running = False
            self.stop_audio()
            if self.ws:
                await self.ws.close()


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Test Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=80, help="Server port")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              Voice Agent Test Client                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Connecting to: ws://{args.host}:{args.port:<5}                          ║
║                                                               ║
║  Speak into your microphone.                                  ║
║  After you stop speaking, you'll hear a beep response.        ║
║                                                               ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
""")

    client = AudioClient(args.host, args.port)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
