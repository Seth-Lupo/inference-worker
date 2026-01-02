#!/usr/bin/env python3
"""
Entry point for the Voice Agent Inference Worker.

Usage:
    python run.py                    # Run on port 80 (requires sudo on Linux/Mac)
    python run.py --port 8080        # Run on port 8080
    python run.py --host 127.0.0.1   # Bind to localhost only
"""
import argparse
import asyncio
import logging
import sys

# Add src to path
sys.path.insert(0, ".")

from src.server import run_server


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Inference Worker")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=80, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # TTS backend configuration
    parser.add_argument(
        "--tts-backend",
        choices=["mock", "cosyvoice"],
        default="mock",
        help="TTS backend to use (default: mock)"
    )
    parser.add_argument(
        "--triton-url",
        default="localhost:8001",
        help="Triton Inference Server gRPC URL (default: localhost:8001)"
    )
    parser.add_argument(
        "--tts-model",
        default="cosyvoice2",
        help="TTS model name on Triton (default: cosyvoice2)"
    )

    args = parser.parse_args()

    setup_logging(args.debug)

    tts_info = f"TTS: {args.tts_backend}"
    if args.tts_backend == "cosyvoice":
        tts_info += f" ({args.triton_url}/{args.tts_model})"

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           Voice Agent Inference Worker                        ║
╠═══════════════════════════════════════════════════════════════╣
║  WebSocket: ws://{args.host}:{args.port:<5}                              ║
║  {tts_info:<61}║
║                                                               ║
║  Connect with a WebSocket client and send audio to test.     ║
║  Audio format: PCM16, 16kHz, mono                             ║
║                                                               ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        asyncio.run(run_server(
            host=args.host,
            port=args.port,
            tts_backend=args.tts_backend,
            triton_url=args.triton_url,
            tts_model=args.tts_model,
        ))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
