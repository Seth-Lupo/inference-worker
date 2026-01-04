"""
Chatterbox S3Gen - Triton Python Backend with torch.compile

Converts speech tokens to audio using:
- Flow decoder: speech tokens -> mel spectrogram
- HiFT vocoder: mel -> audio waveform

Input: Speech tokens + S3Gen conditioning
Output: Audio waveform (24kHz)
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

MODEL_DIR = Path(__file__).parent
# chatterbox_tts module is in the main chatterbox model directory
CHATTERBOX_DIR = MODEL_DIR.parent.parent / "chatterbox" / "1"
sys.path.insert(0, str(CHATTERBOX_DIR))


class TritonPythonModel:
    """S3Gen speech token to audio converter."""

    def initialize(self, args):
        import torch
        from safetensors.torch import load_file

        self.model_config = json.loads(args["model_config"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        # Import chatterbox modules
        from chatterbox_tts import S3Gen, S3GEN_SR, SPEECH_VOCAB_SIZE, drop_invalid_tokens

        # Parameters
        params = self.model_config.get("parameters", {})
        assets_dir = Path(params.get("ASSETS_DIR", {}).get("string_value", "/models/tts/chatterbox_assets"))
        use_compile = params.get("USE_TORCH_COMPILE", {}).get("string_value", "true").lower() == "true"
        self.default_timesteps = int(params.get("DEFAULT_TIMESTEPS", {}).get("string_value", "7"))

        self.logger.log_info("S3Gen: Initializing")
        self.logger.log_info(f"  Assets: {assets_dir}")
        self.logger.log_info(f"  Device: {self.device}")
        self.logger.log_info(f"  Default timesteps: {self.default_timesteps}")

        # Store constants
        self.S3GEN_SR = S3GEN_SR
        self.SPEECH_VOCAB_SIZE = SPEECH_VOCAB_SIZE
        self.drop_invalid_tokens = drop_invalid_tokens

        # Load S3Gen (flow decoder + HiFT vocoder)
        self.logger.log_info("  Loading S3Gen...")
        s3gen_weights = assets_dir / "s3gen.safetensors"
        self.s3gen = S3Gen(use_fp16=True)
        if s3gen_weights.exists():
            self.s3gen.load_state_dict(load_file(str(s3gen_weights)), strict=False)
        self.s3gen = self.s3gen.to(self.device).eval()

        # Apply torch.compile
        if use_compile and hasattr(torch, 'compile'):
            self.logger.log_info("  Applying torch.compile()...")
            try:
                self.s3gen = torch.compile(self.s3gen, mode="reduce-overhead")
                self.logger.log_info("  torch.compile() applied successfully")
            except Exception as e:
                self.logger.log_warn(f"  torch.compile() failed: {e}")
        else:
            self.logger.log_info("  Running in eager mode")

        # Load default conditioning (fallback)
        self.default_cond = None
        conds_path = assets_dir / "conds.pt"
        if conds_path.exists():
            conds_data = torch.load(conds_path, weights_only=True)
            self.default_cond = conds_data.get('gen', conds_data)
            if isinstance(self.default_cond, dict):
                for k, v in self.default_cond.items():
                    if torch.is_tensor(v):
                        self.default_cond[k] = v.to(self.device)
            self.logger.log_info("  Loaded default conditioning")

        # Release PyTorch cached memory to help other models (ONNX) allocate
        torch.cuda.empty_cache()

        self.logger.log_info("S3Gen: Ready")

    def execute(self, requests):
        import torch

        responses = []

        for request in requests:
            try:
                # Get speech tokens
                tokens_tensor = pb_utils.get_input_tensor_by_name(request, "speech_tokens")
                speech_tokens = torch.from_numpy(tokens_tensor.as_numpy().flatten()).to(self.device)

                # Get S3Gen conditioning
                cond_tensor = pb_utils.get_input_tensor_by_name(request, "s3gen_conditioning")
                cond_bytes = cond_tensor.as_numpy().tobytes()
                s3gen_cond = torch.load(io.BytesIO(cond_bytes), weights_only=False)

                # Move conditioning to device
                if isinstance(s3gen_cond, dict):
                    for k, v in s3gen_cond.items():
                        if torch.is_tensor(v):
                            s3gen_cond[k] = v.to(self.device)

                # Get timesteps (optional)
                n_timesteps = self.default_timesteps
                ts_tensor = pb_utils.get_input_tensor_by_name(request, "n_timesteps")
                if ts_tensor is not None:
                    n_timesteps = int(ts_tensor.as_numpy()[0])

                self.logger.log_info(f"S3Gen: {len(speech_tokens)} tokens, {n_timesteps} steps")

                # Clean tokens
                speech_tokens = self.drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < self.SPEECH_VOCAB_SIZE]

                if len(speech_tokens) == 0:
                    audio = np.zeros(0, dtype=np.float32)
                else:
                    with torch.inference_mode():
                        wav, _ = self.s3gen.inference(
                            speech_tokens=speech_tokens,
                            ref_dict=s3gen_cond,
                            n_timesteps=n_timesteps,
                        )
                        audio = wav.squeeze().cpu().numpy().astype(np.float32)

                audio_output = pb_utils.Tensor("audio", audio)
                sr_output = pb_utils.Tensor("sample_rate", np.array([self.S3GEN_SR], dtype=np.int32))
                responses.append(pb_utils.InferenceResponse([audio_output, sr_output]))

            except Exception as e:
                self.logger.log_error(f"S3Gen error: {e}")
                import traceback
                traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self):
        self.logger.log_info("S3Gen: Finalizing")
        import torch
        torch.cuda.empty_cache()
