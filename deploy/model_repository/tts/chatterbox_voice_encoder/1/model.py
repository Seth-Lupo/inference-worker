"""
Chatterbox Voice Encoder - Triton Python Backend

Extracts speaker embedding from reference audio for S3Gen voice cloning.
Returns serialized S3Gen conditioning dict.

Input: Reference audio (16kHz or resampled)
Output: Serialized S3Gen ref_dict (torch.save format)
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
    """Voice encoder for S3Gen conditioning."""

    def initialize(self, args):
        import torch
        from safetensors.torch import load_file

        self.model_config = json.loads(args["model_config"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        # Import chatterbox modules
        from chatterbox_tts import S3Gen, S3GEN_SR

        # Paths
        params = self.model_config.get("parameters", {})
        assets_dir = Path(params.get("ASSETS_DIR", {}).get("string_value", "/models/tts/chatterbox_assets"))

        self.logger.log_info("VoiceEncoder: Initializing")
        self.logger.log_info(f"  Assets: {assets_dir}")
        self.logger.log_info(f"  Device: {self.device}")

        # Store constants
        self.S3GEN_SR = S3GEN_SR

        # Load S3Gen (only need embed_ref method)
        self.logger.log_info("  Loading S3Gen for embedding...")
        s3gen_weights = assets_dir / "s3gen.safetensors"
        self.s3gen = S3Gen(use_fp16=True)
        if s3gen_weights.exists():
            self.s3gen.load_state_dict(load_file(str(s3gen_weights)), strict=False)
        self.s3gen = self.s3gen.to(self.device).eval()

        # Load default conditioning (for fallback)
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

        self.logger.log_info("VoiceEncoder: Ready")

    def execute(self, requests):
        import torch
        import librosa

        responses = []

        for request in requests:
            try:
                # Get reference audio
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "reference_audio")
                audio = audio_tensor.as_numpy().flatten().astype(np.float32)

                # Get sample rate (default 16000)
                sr_tensor = pb_utils.get_input_tensor_by_name(request, "sample_rate")
                sample_rate = 16000
                if sr_tensor is not None:
                    sample_rate = int(sr_tensor.as_numpy()[0])

                # Normalize audio
                if np.abs(audio).max() > 1.0:
                    audio = audio / 32768.0

                self.logger.log_info(f"VoiceEncoder: Processing {len(audio)} samples @ {sample_rate}Hz")

                # Resample to S3Gen sample rate
                if sample_rate != self.S3GEN_SR:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.S3GEN_SR)

                # Extract S3Gen conditioning
                with torch.inference_mode():
                    s3gen_cond = self.s3gen.embed_ref(audio, self.S3GEN_SR)

                # Serialize to bytes
                buffer = io.BytesIO()
                torch.save(s3gen_cond, buffer)
                cond_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

                output = pb_utils.Tensor("s3gen_conditioning", cond_bytes)
                responses.append(pb_utils.InferenceResponse([output]))

            except Exception as e:
                self.logger.log_error(f"VoiceEncoder error: {e}")
                import traceback
                traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self):
        self.logger.log_info("VoiceEncoder: Finalizing")
        import torch
        torch.cuda.empty_cache()
