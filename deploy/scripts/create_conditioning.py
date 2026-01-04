#!/usr/bin/env python3
"""
Create pre-computed conditioning.pt files for T3 vLLM backend.

Supports:
- Single voice from reference audio
- Multiple voices from a directory
- Default voice from conds.pt

Usage:
    # Compile all voices from a directory:
    python create_conditioning.py --voices-dir ../voices --output-dir ../models/t3_weights/voices

    # Single voice from reference audio:
    python create_conditioning.py --reference-audio voice.wav --output ../models/t3_weights/voices/my_voice.pt

    # Default voice from conds.pt:
    python create_conditioning.py --assets-dir ../model_repository/tts/chatterbox_assets
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file

# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def load_t3_conditioning_encoder(assets_dir: Path, device: torch.device):
    """Load T3 conditioning encoder from weights."""
    # Add chatterbox_tts to path
    model_dir = assets_dir.parent / "chatterbox" / "1"
    sys.path.insert(0, str(model_dir))

    from chatterbox_tts import T3Config, T3CondEnc, LearnedPositionEmbeddings

    t3_weights_path = assets_dir / "t3_cfg.safetensors"
    if not t3_weights_path.exists():
        raise FileNotFoundError(f"T3 weights not found: {t3_weights_path}")

    t3_weights = load_file(str(t3_weights_path))
    config = T3Config()

    # Load conditioning encoder
    cond_enc = T3CondEnc(config)
    cond_enc.load_state_dict({
        k.replace('cond_enc.', ''): v
        for k, v in t3_weights.items() if k.startswith('cond_enc.')
    })
    cond_enc = cond_enc.to(device).eval()

    # Load speech embeddings
    speech_emb = torch.nn.Embedding(
        config.speech_tokens_dict_size,
        config.n_channels
    )
    speech_emb.load_state_dict({
        k.replace('speech_emb.', ''): v
        for k, v in t3_weights.items() if k.startswith('speech_emb.')
    })
    speech_emb = speech_emb.to(device).eval()

    # Load position embeddings
    speech_pos_emb = LearnedPositionEmbeddings(
        config.max_speech_tokens + 2 + 2,
        config.n_channels
    )
    speech_pos_emb.load_state_dict({
        k.replace('speech_pos_emb.', ''): v
        for k, v in t3_weights.items() if k.startswith('speech_pos_emb.')
    })
    speech_pos_emb = speech_pos_emb.to(device).eval()

    return config, cond_enc, speech_emb, speech_pos_emb


def load_s3gen_and_voice_encoder(assets_dir: Path, device: torch.device):
    """Load S3Gen tokenizer and voice encoder."""
    model_dir = assets_dir.parent / "chatterbox" / "1"
    sys.path.insert(0, str(model_dir))

    from chatterbox_tts import S3Gen, VoiceEncoder

    # Load voice encoder
    ve_weights = assets_dir / "ve.safetensors"
    voice_encoder = VoiceEncoder()
    if ve_weights.exists():
        voice_encoder.load_state_dict(load_file(str(ve_weights)))
    voice_encoder = voice_encoder.to(device).eval()

    # Load S3Gen (for tokenizer)
    s3gen_weights = assets_dir / "s3gen.safetensors"
    s3gen = S3Gen(use_fp16=False)
    if s3gen_weights.exists():
        s3gen.load_state_dict(load_file(str(s3gen_weights)), strict=False)
    s3gen = s3gen.to(device).eval()

    return s3gen, voice_encoder


def create_conditioning_from_conds(
    conds_path: Path,
    cond_enc,
    speech_emb,
    speech_pos_emb,
    device: torch.device
) -> torch.Tensor:
    """Create conditioning tensor from conds.pt file."""
    model_dir = conds_path.parent.parent / "chatterbox" / "1"
    sys.path.insert(0, str(model_dir))
    from chatterbox_tts import T3Cond

    conds_data = torch.load(conds_path, weights_only=True)
    t3_cond = T3Cond(**conds_data['t3']).to(device)

    # Compute speech embeddings
    cond_prompt_speech_emb = (
        speech_emb(t3_cond.cond_prompt_speech_tokens)[0] +
        speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
    )

    # Create full conditioning
    full_cond = T3Cond(
        speaker_emb=t3_cond.speaker_emb,
        cond_prompt_speech_tokens=t3_cond.cond_prompt_speech_tokens,
        cond_prompt_speech_emb=cond_prompt_speech_emb,
        emotion_adv=t3_cond.emotion_adv
    ).to(device)

    # Encode
    with torch.inference_mode():
        conditioning = cond_enc(full_cond)

    return conditioning


def create_conditioning_from_audio(
    audio_path: Path,
    config,
    cond_enc,
    speech_emb,
    speech_pos_emb,
    s3gen,
    voice_encoder,
    device: torch.device,
    emotion_adv: float = 0.5
) -> torch.Tensor:
    """Create conditioning tensor from reference audio."""
    import librosa
    import numpy as np

    # Import after adding to path
    from chatterbox_tts import T3Cond, S3_SR

    # Load audio at 16kHz
    audio, sr = librosa.load(str(audio_path), sr=16000)
    audio = audio.astype(np.float32)

    print(f"  Audio loaded: {len(audio)/16000:.1f}s @ 16kHz")

    # Get speaker embedding from voice encoder
    ve_embed = torch.from_numpy(
        voice_encoder.embeds_from_wavs([audio], sample_rate=S3_SR)
    ).mean(axis=0, keepdim=True).to(device)

    # Get speech tokens from reference (first 6 seconds)
    ENC_COND_LEN = 6 * S3_SR
    ref_wav = torch.from_numpy(audio[:ENC_COND_LEN]).unsqueeze(0).to(device)

    cond_prompt_tokens, _ = s3gen.tokenizer.forward(
        ref_wav,
        max_len=config.speech_cond_prompt_len
    )
    cond_prompt_tokens = torch.atleast_2d(cond_prompt_tokens)

    print(f"  Speaker embedding: {ve_embed.shape}")
    print(f"  Prompt tokens: {cond_prompt_tokens.shape}")

    # Compute speech embeddings
    cond_prompt_speech_emb = (
        speech_emb(cond_prompt_tokens)[0] +
        speech_pos_emb(cond_prompt_tokens)
    )

    # Create conditioning
    t3_cond = T3Cond(
        speaker_emb=ve_embed,
        cond_prompt_speech_tokens=cond_prompt_tokens,
        cond_prompt_speech_emb=cond_prompt_speech_emb,
        emotion_adv=emotion_adv * torch.ones(1, 1).to(device)
    ).to(device)

    # Encode
    with torch.inference_mode():
        conditioning = cond_enc(t3_cond)

    return conditioning


def process_voices_directory(
    voices_dir: Path,
    output_dir: Path,
    assets_dir: Path,
    device: torch.device,
    emotion_adv: float = 0.5
) -> Dict[str, str]:
    """Process all voice files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(voices_dir.glob(f"*{ext}"))
        audio_files.extend(voices_dir.glob(f"*{ext.upper()}"))

    if not audio_files:
        print(f"No audio files found in {voices_dir}")
        return {}

    print(f"Found {len(audio_files)} voice files in {voices_dir}")

    # Load models once
    print("Loading T3 conditioning encoder...")
    config, cond_enc, speech_emb, speech_pos_emb = load_t3_conditioning_encoder(
        assets_dir, device
    )

    print("Loading S3Gen and VoiceEncoder...")
    s3gen, voice_encoder = load_s3gen_and_voice_encoder(assets_dir, device)

    # Process each voice
    voices = {}
    for audio_path in sorted(audio_files):
        voice_name = audio_path.stem  # filename without extension
        output_path = output_dir / f"{voice_name}.pt"

        print(f"\nProcessing voice: {voice_name}")
        print(f"  Source: {audio_path}")

        try:
            conditioning = create_conditioning_from_audio(
                audio_path,
                config,
                cond_enc,
                speech_emb,
                speech_pos_emb,
                s3gen,
                voice_encoder,
                device,
                emotion_adv
            )

            torch.save(conditioning.cpu(), output_path)
            print(f"  Saved: {output_path}")
            print(f"  Shape: {conditioning.shape}")

            voices[voice_name] = str(output_path.name)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return voices


def main():
    parser = argparse.ArgumentParser(description="Create T3 voice conditioning files")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).parent.parent / "model_repository/tts/chatterbox_assets",
        help="Path to chatterbox_assets directory"
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        help="Directory containing voice audio files to process"
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Single reference audio file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for single voice conditioning.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for multiple voices (default: models/t3_weights/voices)"
    )
    parser.add_argument(
        "--emotion",
        type=float,
        default=0.5,
        help="Emotion/exaggeration level (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--default-voice",
        type=str,
        default="default",
        help="Name for the default voice (from conds.pt)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assets_dir = args.assets_dir.resolve()
    print(f"Assets directory: {assets_dir}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = assets_dir.parent.parent.parent / "models/t3_weights/voices"

    voices = {}

    # Mode 1: Process voices directory
    if args.voices_dir:
        voices_dir = args.voices_dir.resolve()
        print(f"\n=== Processing voices directory: {voices_dir} ===")
        voices.update(process_voices_directory(
            voices_dir, output_dir, assets_dir, device, args.emotion
        ))

    # Mode 2: Single reference audio
    elif args.reference_audio:
        audio_path = args.reference_audio.resolve()
        voice_name = audio_path.stem

        if args.output:
            output_path = args.output.resolve()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{voice_name}.pt"

        print(f"\n=== Processing single voice: {voice_name} ===")

        config, cond_enc, speech_emb, speech_pos_emb = load_t3_conditioning_encoder(
            assets_dir, device
        )
        s3gen, voice_encoder = load_s3gen_and_voice_encoder(assets_dir, device)

        conditioning = create_conditioning_from_audio(
            audio_path,
            config,
            cond_enc,
            speech_emb,
            speech_pos_emb,
            s3gen,
            voice_encoder,
            device,
            args.emotion
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(conditioning.cpu(), output_path)
        print(f"Saved: {output_path}")

        voices[voice_name] = str(output_path.name)

    # Mode 3: Default from conds.pt (always include if available)
    conds_path = assets_dir / "conds.pt"
    if conds_path.exists():
        print(f"\n=== Creating default voice from conds.pt ===")

        config, cond_enc, speech_emb, speech_pos_emb = load_t3_conditioning_encoder(
            assets_dir, device
        )

        conditioning = create_conditioning_from_conds(
            conds_path, cond_enc, speech_emb, speech_pos_emb, device
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        default_path = output_dir / f"{args.default_voice}.pt"
        torch.save(conditioning.cpu(), default_path)
        print(f"Saved default voice: {default_path}")

        voices[args.default_voice] = str(default_path.name)

        # Also save as conditioning.pt for backwards compatibility
        compat_path = output_dir.parent / "conditioning.pt"
        torch.save(conditioning.cpu(), compat_path)
        print(f"Saved compatibility file: {compat_path}")

    # Save manifest
    if voices:
        manifest_path = output_dir / "voices.json"
        manifest = {
            "voices": voices,
            "default": args.default_voice if args.default_voice in voices else list(voices.keys())[0]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n=== Voice manifest saved: {manifest_path} ===")
        print(json.dumps(manifest, indent=2))

    print(f"\n=== Done! {len(voices)} voice(s) created ===")


if __name__ == "__main__":
    main()
