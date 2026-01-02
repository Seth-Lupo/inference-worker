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
    args = parser.parse_args()

    setup_logging(args.debug)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           Voice Agent Inference Worker                        ║
╠═══════════════════════════════════════════════════════════════╣
║  WebSocket: ws://{args.host}:{args.port:<5}                              ║
║                                                               ║
║  Connect with a WebSocket client and send audio to test.     ║
║  Audio format: PCM16, 16kHz, mono                             ║
║                                                               ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
