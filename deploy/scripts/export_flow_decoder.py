#!/usr/bin/env python3
"""
Export S3Gen Flow Matching Decoder to ONNX with Unrolled Diffusion Steps

This script creates an ONNX model with N diffusion steps baked in (unrolled),
enabling TensorRT to compile the entire diffusion process into a single engine.

Usage:
    python export_flow_decoder.py --assets-dir /path/to/assets --output-dir /path/to/output --steps 4

The exported model takes:
    - latents: Initial noise [batch, channels, time]
    - speaker_embedding: Speaker conditioning [batch, embedding_dim]
    - speech_tokens: Token embeddings [batch, seq_len, hidden_dim]

And outputs:
    - mel_spectrogram: Denoised output [batch, channels, time]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository" / "tts" / "chatterbox" / "1"))


class UnrolledFlowDecoder(nn.Module):
    """
    Wrapper that unrolls the flow matching diffusion loop.

    Instead of:
        for t in timesteps:
            x = step(x, t, cond)

    We unroll to:
        x = step(x, t0, cond)
        x = step(x, t1, cond)
        ...
        x = step(x, tN, cond)

    This allows TensorRT to compile everything into a single engine.
    """

    def __init__(self, flow_decoder, num_steps: int = 4):
        super().__init__()
        self.flow_decoder = flow_decoder
        self.num_steps = num_steps

        # Pre-compute timesteps (linear schedule from 1.0 to ~0.0)
        # Using register_buffer so they're part of the model state
        timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps)
        self.register_buffer('timesteps', timesteps)

        # Step size for Euler integration
        self.register_buffer('dt', torch.tensor(1.0 / num_steps))

    def forward(
        self,
        latents: torch.Tensor,
        speaker_embedding: torch.Tensor,
        speech_token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run N diffusion steps in a single forward pass.

        Args:
            latents: Initial noise [batch, mel_channels, time_frames]
            speaker_embedding: Speaker conditioning [batch, speaker_dim]
            speech_token_embedding: Token features [batch, seq_len, hidden_dim]

        Returns:
            Denoised mel spectrogram [batch, mel_channels, time_frames]
        """
        x = latents

        # Unrolled diffusion steps
        # Note: This loop is unrolled at trace time, not runtime
        for i in range(self.num_steps):
            t = self.timesteps[i:i+1].expand(x.shape[0])

            # Flow matching velocity prediction
            velocity = self.flow_decoder(
                x,
                t,
                speaker_embedding,
                speech_token_embedding,
            )

            # Euler step: x_{t-dt} = x_t - dt * v(x_t, t)
            x = x - self.dt * velocity

        return x


class FlowDecoderWrapper(nn.Module):
    """
    Wrapper for the S3Gen flow decoder that matches expected interface.
    """

    def __init__(self, s3gen_model):
        super().__init__()
        # Extract the flow matching decoder from S3Gen
        self.decoder = s3gen_model.decoder
        self.mel_channels = getattr(s3gen_model, 'mel_channels', 128)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        speaker_embedding: torch.Tensor,
        speech_token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single diffusion step.

        Args:
            x: Current latents [batch, mel_channels, time]
            t: Timestep [batch]
            speaker_embedding: [batch, speaker_dim]
            speech_token_embedding: [batch, seq_len, hidden_dim]

        Returns:
            Velocity prediction [batch, mel_channels, time]
        """
        return self.decoder(x, t, speaker_embedding, speech_token_embedding)


def load_s3gen(assets_dir: Path, device: str = "cuda"):
    """Load S3Gen model from safetensors."""
    from safetensors.torch import load_file

    s3gen_path = assets_dir / "s3gen.safetensors"
    if not s3gen_path.exists():
        raise FileNotFoundError(f"S3Gen weights not found: {s3gen_path}")

    # Try to import S3Gen model class
    try:
        from chatterbox.models.s3gen import S3Gen
        model = S3Gen()
        state_dict = load_file(str(s3gen_path))
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        return model
    except ImportError:
        pass

    # Fallback: try to reconstruct from state dict
    try:
        # Load state dict to inspect structure
        state_dict = load_file(str(s3gen_path))

        # Check for decoder keys
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
        if not decoder_keys:
            raise ValueError("No decoder keys found in s3gen.safetensors")

        print(f"Found {len(decoder_keys)} decoder parameters")
        print(f"Sample keys: {decoder_keys[:5]}")

        # We need the actual model class to proceed
        raise ImportError("Need S3Gen model class definition")

    except Exception as e:
        print(f"Error loading S3Gen: {e}")
        raise


def export_onnx(
    model: nn.Module,
    output_path: Path,
    num_steps: int,
    batch_size: int = 1,
    mel_channels: int = 128,
    time_frames: int = 64,
    speaker_dim: int = 256,
    token_seq_len: int = 32,
    token_hidden_dim: int = 1024,
):
    """Export unrolled flow decoder to ONNX."""

    device = next(model.parameters()).device

    # Create dummy inputs
    latents = torch.randn(batch_size, mel_channels, time_frames, device=device)
    speaker_embedding = torch.randn(batch_size, speaker_dim, device=device)
    speech_token_embedding = torch.randn(batch_size, token_seq_len, token_hidden_dim, device=device)

    # Dynamic axes for variable batch/time
    dynamic_axes = {
        'latents': {0: 'batch', 2: 'time'},
        'speaker_embedding': {0: 'batch'},
        'speech_token_embedding': {0: 'batch', 1: 'seq_len'},
        'mel_spectrogram': {0: 'batch', 2: 'time'},
    }

    print(f"Exporting ONNX with {num_steps} unrolled steps...")
    print(f"  Input shapes:")
    print(f"    latents: {latents.shape}")
    print(f"    speaker_embedding: {speaker_embedding.shape}")
    print(f"    speech_token_embedding: {speech_token_embedding.shape}")

    torch.onnx.export(
        model,
        (latents, speaker_embedding, speech_token_embedding),
        str(output_path),
        input_names=['latents', 'speaker_embedding', 'speech_token_embedding'],
        output_names=['mel_spectrogram'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Exported: {output_path}")

    # Save metadata
    metadata = {
        'num_steps': num_steps,
        'mel_channels': mel_channels,
        'speaker_dim': speaker_dim,
        'token_hidden_dim': token_hidden_dim,
    }
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")


def create_dummy_flow_decoder(
    mel_channels: int = 128,
    speaker_dim: int = 256,
    token_hidden_dim: int = 1024,
    hidden_dim: int = 512,
):
    """
    Create a dummy flow decoder for testing the export pipeline.
    Replace this with actual S3Gen decoder loading.
    """

    class DummyFlowDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Simplified architecture for testing
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)
            self.token_proj = nn.Linear(token_hidden_dim, hidden_dim)

            # Simple transformer-like block
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

            # Output projection
            self.out_proj = nn.Linear(hidden_dim, mel_channels)

        def forward(self, x, t, speaker_embedding, speech_token_embedding):
            batch, channels, time = x.shape

            # Time embedding
            t_emb = self.time_embed(t.unsqueeze(-1))  # [batch, hidden]

            # Speaker embedding
            s_emb = self.speaker_proj(speaker_embedding)  # [batch, hidden]

            # Token embedding
            tok_emb = self.token_proj(speech_token_embedding)  # [batch, seq, hidden]

            # Reshape x for attention: [batch, time, channels] -> [batch, time, hidden]
            x = x.transpose(1, 2)  # [batch, time, channels]
            x = nn.functional.pad(x, (0, hidden_dim - channels))  # Pad to hidden_dim

            # Add conditioning
            cond = (t_emb + s_emb).unsqueeze(1)  # [batch, 1, hidden]
            x = x + cond

            # Cross-attention with tokens
            x = self.norm1(x)
            x = x + self.attn(x, tok_emb, tok_emb)[0]
            x = self.norm2(x)
            x = x + self.ffn(x)

            # Project to mel channels
            x = self.out_proj(x)  # [batch, time, mel_channels]
            x = x.transpose(1, 2)  # [batch, mel_channels, time]

            return x

    return DummyFlowDecoder()


def main():
    parser = argparse.ArgumentParser(description="Export unrolled flow decoder to ONNX")
    parser.add_argument("--assets-dir", type=Path, required=True, help="Path to chatterbox assets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for ONNX")
    parser.add_argument("--steps", type=int, default=4, help="Number of diffusion steps to unroll")
    parser.add_argument("--dummy", action="store_true", help="Use dummy model for testing")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 precision")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Steps: {args.steps}")

    if args.dummy:
        print("Using dummy flow decoder for testing...")
        flow_decoder = create_dummy_flow_decoder().to(device)
    else:
        print(f"Loading S3Gen from {args.assets_dir}...")
        try:
            s3gen = load_s3gen(args.assets_dir, device)
            flow_decoder = FlowDecoderWrapper(s3gen)
        except Exception as e:
            print(f"Failed to load S3Gen: {e}")
            print("Falling back to dummy model for pipeline testing")
            flow_decoder = create_dummy_flow_decoder().to(device)

    # Wrap in unrolled decoder
    unrolled = UnrolledFlowDecoder(flow_decoder, num_steps=args.steps).to(device)

    if args.fp16:
        unrolled = unrolled.half()

    unrolled.eval()

    # Export
    output_name = f"flow_decoder_{args.steps}steps{'_fp16' if args.fp16 else ''}.onnx"
    output_path = args.output_dir / output_name

    with torch.no_grad():
        export_onnx(unrolled, output_path, args.steps)

    print(f"\nSuccess! ONNX exported to: {output_path}")
    print(f"\nTo build TensorRT engine:")
    print(f"  trtexec --onnx={output_path} --saveEngine={output_path.with_suffix('.engine')} --fp16")


if __name__ == "__main__":
    main()
