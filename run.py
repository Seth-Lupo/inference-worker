#!/usr/bin/env python3
"""
Entry point for the Voice Agent Inference Worker.

Usage:
    python run.py                    # Run on port 80 (requires sudo on Linux/Mac)
    python run.py --port 8080        # Run on port 8080
    python run.py --host 127.0.0.1   # Bind to localhost only

Environment variables:
    TTS_BACKEND           - "mock" or "chatterbox" (default: chatterbox)
    TRITON_URL            - Triton gRPC URL (default: localhost:8001)
    TTS_MODEL             - TTS model name (default: chatterbox)
    VLLM_URL              - vLLM API URL (default: http://localhost:8000)
    REFERENCE_AUDIO_PATH  - Path to reference audio for voice cloning
    REFERENCE_TEXT        - Text spoken in reference audio
    LOG_LEVEL             - DEBUG, INFO, WARNING, ERROR (default: INFO)
"""
import argparse
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.insert(0, ".")

from src.server import run_server


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Inference Worker")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=80, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # TTS config from env vars with CLI override
    parser.add_argument(
        "--tts-backend",
        choices=["mock", "chatterbox"],
        default=os.environ.get("TTS_BACKEND", "chatterbox"),
        help="TTS backend (env: TTS_BACKEND, default: chatterbox)"
    )
    parser.add_argument(
        "--triton-url",
        default=os.environ.get("TRITON_URL", "localhost:8001"),
        help="Triton gRPC URL (env: TRITON_URL, default: localhost:8001)"
    )
    parser.add_argument(
        "--tts-model",
        default=os.environ.get("TTS_MODEL", "chatterbox"),
        help="TTS model name (env: TTS_MODEL, default: chatterbox)"
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_URL", "http://localhost:8000"),
        help="vLLM API URL (env: VLLM_URL, default: http://localhost:8000)"
    )
    parser.add_argument(
        "--reference-audio",
        default=os.environ.get("REFERENCE_AUDIO_PATH"),
        help="Path to reference audio for voice cloning (env: REFERENCE_AUDIO_PATH)"
    )
    parser.add_argument(
        "--reference-text",
        default=os.environ.get("REFERENCE_TEXT"),
        help="Text spoken in reference audio (env: REFERENCE_TEXT)"
    )

    args = parser.parse_args()

    # Log level from env or --debug flag
    log_level = "DEBUG" if args.debug else os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    tts_info = f"TTS: {args.tts_backend}"
    if args.tts_backend == "chatterbox":
        tts_info += f" ({args.triton_url}/{args.tts_model})"

    voice_info = "Voice: default"
    if args.reference_audio:
        voice_info = f"Voice: {os.path.basename(args.reference_audio)}"

    print(f"""
+---------------------------------------------------------------+
|           Voice Agent Inference Worker                        |
+---------------------------------------------------------------+
|  WebSocket: ws://{args.host}:{args.port:<5}                              |
|  {tts_info:<61}|
|  vLLM: {args.vllm_url:<55}|
|  {voice_info:<61}|
|                                                               |
|  Connect with a WebSocket client and send audio to test.     |
|  Audio format: PCM16, 16kHz, mono                             |
|                                                               |
|  Press Ctrl+C to stop                                         |
+---------------------------------------------------------------+
""")

    try:
        asyncio.run(run_server(
            host=args.host,
            port=args.port,
            tts_backend=args.tts_backend,
            triton_url=args.triton_url,
            tts_model=args.tts_model,
            reference_audio_path=args.reference_audio,
            reference_text=args.reference_text,
        ))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
