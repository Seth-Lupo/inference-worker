"""
Chatterbox TTS - BLS Orchestrator

Architecture:
  chatterbox (this model - BLS orchestrator)
    ├── chatterbox_voice_encoder (Python backend) - reference audio -> S3Gen conditioning
    ├── t3 (vLLM backend via gRPC) - text + T3 conditioning -> speech tokens
    └── chatterbox_s3gen (Python backend + torch.compile) - tokens + conditioning -> audio

This orchestrator:
1. Calls voice_encoder if reference audio provided (or uses default conditioning)
2. Calls T3 via gRPC for speech token generation
3. Calls s3gen for audio synthesis with progressive streaming

Benefits of BLS architecture:
- Each component can be independently scaled
- Voice encoder can be shared/cached
- S3Gen can use torch.compile() optimizations
- Clean separation of concerns
"""
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils

MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))


class TritonPythonModel:
    """Chatterbox TTS BLS Orchestrator."""

    def initialize(self, args: dict):
        import torch
        import tritonclient.grpc as grpcclient

        self.model_config = json.loads(args["model_config"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        # Import only what we need for orchestration (not the heavy models)
        from chatterbox_tts import SPEECH_VOCAB_SIZE, SPEECH_TOKEN_OFFSET
        from chatterbox_tts import punc_norm

        self.SPEECH_VOCAB_SIZE = SPEECH_VOCAB_SIZE
        self.SPEECH_TOKEN_OFFSET = SPEECH_TOKEN_OFFSET
        self.punc_norm = punc_norm

        self.logger.log_info("Chatterbox BLS: Initializing")

        # Paths for pre-compiled voices (T3 conditioning)
        voices_dir = Path("/models/t3_weights/voices")
        assets_dir = MODEL_DIR.parent / "chatterbox_assets"

        # Load pre-compiled voice conditionings (for T3)
        self.voices = {}
        self.default_voice = "default"

        manifest_path = voices_dir / "voices.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.default_voice = manifest.get("default", "default")

            for voice_name, voice_file in manifest.get("voices", {}).items():
                voice_path = voices_dir / voice_file
                if voice_path.exists():
                    voice_data = torch.load(voice_path, weights_only=True)
                    if isinstance(voice_data, dict) and 't3' in voice_data:
                        self.voices[voice_name] = voice_data['t3'].to(self.device)
                    else:
                        self.voices[voice_name] = voice_data.to(self.device)
                    self.logger.log_info(f"  Loaded voice: {voice_name}")

        # Fallback: load conditioning.pt directly
        if not self.voices:
            compat_path = voices_dir.parent / "conditioning.pt"
            if compat_path.exists():
                conditioning = torch.load(compat_path, weights_only=True)
                self.voices["default"] = conditioning.to(self.device)
                self.logger.log_info("  Loaded fallback voice: default")

        self.logger.log_info(f"  Available voices: {list(self.voices.keys())}")

        # Load default S3Gen conditioning (for when no reference audio)
        self.default_s3gen_cond_bytes = None
        conds_path = assets_dir / "conds.pt"
        if conds_path.exists():
            conds_data = torch.load(conds_path, weights_only=True)
            default_cond = conds_data.get('gen', conds_data)
            # Serialize to bytes for passing to s3gen
            buffer = io.BytesIO()
            torch.save(default_cond, buffer)
            self.default_s3gen_cond_bytes = buffer.getvalue()
            self.logger.log_info("  Loaded default S3Gen conditioning")

        # Triton client for T3 (vLLM backend - must use gRPC)
        triton_url = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
        self.t3_model_name = os.environ.get("T3_MODEL_NAME", "t3")
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self.logger.log_info(f"  T3 via gRPC: {triton_url}/{self.t3_model_name}")

        # Generation parameters
        self.temperature = float(os.environ.get("TTS_TEMPERATURE", "0.8"))
        self.top_p = float(os.environ.get("TTS_TOP_P", "0.8"))

        # Progressive streaming schedule
        chunk_str = os.environ.get("TTS_PROGRESSIVE_CHUNKS", "4,8,16,32,32,32,32,32")
        self.progressive_chunks = [int(x) for x in chunk_str.split(",")]
        steps_str = os.environ.get("TTS_PROGRESSIVE_STEPS", "2,3,5,7,7,7,7,7")
        self.progressive_steps = [int(x) for x in steps_str.split(",")]
        self.steady_chunk_size = int(os.environ.get("TTS_CHUNK_TOKENS", "32"))
        self.steady_diffusion_steps = int(os.environ.get("S3GEN_DIFFUSION_STEPS", "7"))

        # Decoupled mode for streaming
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.logger.log_info(f"Chatterbox BLS: Ready (decoupled={self.decoupled})")
        self.logger.log_info(f"  Progressive chunks: {self.progressive_chunks}")
        self.logger.log_info(f"  Progressive steps:  {self.progressive_steps}")

    def _get_voice_conditioning(self, voice_id: Optional[str] = None) -> "torch.Tensor":
        """Get T3 conditioning tensor for the specified voice."""
        if voice_id is None or voice_id not in self.voices:
            voice_id = self.default_voice

        if voice_id not in self.voices:
            if self.voices:
                voice_id = list(self.voices.keys())[0]
            else:
                raise RuntimeError("No voices loaded")

        return self.voices[voice_id]

    def _call_voice_encoder(self, ref_audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Call chatterbox_voice_encoder via BLS to get S3Gen conditioning."""
        # Prepare inputs
        audio_tensor = pb_utils.Tensor("reference_audio", ref_audio.astype(np.float32))
        sr_tensor = pb_utils.Tensor("sample_rate", np.array([sample_rate], dtype=np.int32))

        # Create BLS request
        infer_request = pb_utils.InferenceRequest(
            model_name="chatterbox_voice_encoder",
            requested_output_names=["s3gen_conditioning"],
            inputs=[audio_tensor, sr_tensor]
        )

        # Execute synchronously
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise RuntimeError(f"VoiceEncoder error: {infer_response.error().message()}")

        # Get serialized conditioning
        cond_tensor = pb_utils.get_output_tensor_by_name(infer_response, "s3gen_conditioning")
        return cond_tensor.as_numpy().tobytes()

    def _call_t3(self, text: str, conditioning: "torch.Tensor") -> List[int]:
        """Call T3 via gRPC (vLLM backend)."""
        import base64
        import torch
        import tritonclient.grpc as grpcclient

        # Prepare text input
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([text.encode()], dtype=object))

        # Serialize conditioning tensor
        buffer = io.BytesIO()
        torch.save(conditioning.cpu(), buffer)
        cond_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        image_input = grpcclient.InferInput("image", [1], "BYTES")
        image_input.set_data_from_numpy(np.array([cond_b64.encode()], dtype=object))

        # Sampling parameters
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": 1000,
            "ignore_eos": False,
        }
        params_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
        params_input.set_data_from_numpy(
            np.array([json.dumps(sampling_params).encode()], dtype=object)
        )

        text_output = grpcclient.InferRequestedOutput("text_output")

        try:
            result = self.triton_client.infer(
                model_name=self.t3_model_name,
                inputs=[text_input, image_input, params_input],
                outputs=[text_output],
            )
            output = result.as_numpy("text_output")
            return self._parse_tokens(output)
        except Exception as e:
            self.logger.log_error(f"T3 call failed: {e}")
            raise

    def _parse_tokens(self, output: np.ndarray) -> List[int]:
        """Parse speech tokens from T3 output."""
        text = output[0].decode() if isinstance(output[0], bytes) else str(output[0])

        tokens = []
        for part in text.split():
            try:
                token_id = int(part) - self.SPEECH_TOKEN_OFFSET
                if 0 <= token_id < self.SPEECH_VOCAB_SIZE:
                    tokens.append(token_id)
            except ValueError:
                pass
        return tokens

    def _call_s3gen(self, speech_tokens: List[int], s3gen_cond_bytes: bytes, n_timesteps: int = 7) -> np.ndarray:
        """Call chatterbox_s3gen via BLS to convert tokens to audio."""
        # Prepare inputs
        tokens_tensor = pb_utils.Tensor("speech_tokens", np.array(speech_tokens, dtype=np.int64))
        cond_tensor = pb_utils.Tensor("s3gen_conditioning", np.frombuffer(s3gen_cond_bytes, dtype=np.uint8))
        steps_tensor = pb_utils.Tensor("n_timesteps", np.array([n_timesteps], dtype=np.int32))

        # Create BLS request
        infer_request = pb_utils.InferenceRequest(
            model_name="chatterbox_s3gen",
            requested_output_names=["audio", "sample_rate"],
            inputs=[tokens_tensor, cond_tensor, steps_tensor]
        )

        # Execute synchronously
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise RuntimeError(f"S3Gen error: {infer_response.error().message()}")

        # Get audio output
        audio_tensor = pb_utils.get_output_tensor_by_name(infer_response, "audio")
        return audio_tensor.as_numpy().flatten().astype(np.float32)

    def execute(self, requests: List) -> List:
        """Process TTS requests via BLS orchestration."""
        import torch

        responses = []

        for request in requests:
            request_id = request.request_id() or "unknown"
            start_time = time.time()

            try:
                # Get text input
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = text_tensor.as_numpy()[0][0].decode("utf-8")

                # Get optional reference audio
                ref_tensor = pb_utils.get_input_tensor_by_name(request, "reference_audio")
                ref_audio = None
                if ref_tensor is not None:
                    ref_audio = ref_tensor.as_numpy().flatten().astype(np.float32)
                    # Normalize
                    if np.abs(ref_audio).max() > 1.0:
                        ref_audio = ref_audio / 32768.0

                # Get voice_id (optional)
                voice_tensor = pb_utils.get_input_tensor_by_name(request, "voice_id")
                voice_id = None
                if voice_tensor is not None:
                    voice_id = voice_tensor.as_numpy()[0].decode("utf-8")

                self.logger.log_info(
                    f"[{request_id}] Text: '{text[:50]}...' voice={voice_id or self.default_voice}"
                )

                # Step 1: Get T3 voice conditioning
                t3_conditioning = self._get_voice_conditioning(voice_id)

                # Step 2: Get S3Gen conditioning (via BLS or default)
                if ref_audio is not None:
                    self.logger.log_info(f"[{request_id}] Calling voice_encoder for reference audio")
                    s3gen_cond_bytes = self._call_voice_encoder(ref_audio)
                else:
                    if self.default_s3gen_cond_bytes is None:
                        raise RuntimeError("No S3Gen conditioning available")
                    s3gen_cond_bytes = self.default_s3gen_cond_bytes

                # Step 3: Prepare text and call T3
                text_prompt = "[START]" + self.punc_norm(text) + "[STOP]"

                if self.decoupled:
                    sender = request.get_response_sender()
                    self._generate_streaming(
                        sender, request_id, text_prompt,
                        t3_conditioning, s3gen_cond_bytes, start_time
                    )
                else:
                    # Non-streaming mode
                    tokens = self._call_t3(text_prompt, t3_conditioning)
                    audio = self._call_s3gen(tokens, s3gen_cond_bytes)
                    tensor = pb_utils.Tensor("audio", audio)
                    responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))

            except Exception as e:
                self.logger.log_error(f"[{request_id}] Error: {e}")
                import traceback
                traceback.print_exc()

                if self.decoupled:
                    sender = request.get_response_sender()
                    sender.send(
                        pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e))),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    ))

        return responses if not self.decoupled else None

    def _generate_streaming(
        self,
        sender,
        request_id: str,
        text_prompt: str,
        t3_conditioning: "torch.Tensor",
        s3gen_cond_bytes: bytes,
        start_time: float
    ):
        """Generate with progressive streaming via BLS calls to s3gen."""
        # Call T3 for all tokens
        tokens = self._call_t3(text_prompt, t3_conditioning)

        token_gen_time = time.time() - start_time
        self.logger.log_info(
            f"[{request_id}] Generated {len(tokens)} tokens in {token_gen_time:.2f}s"
        )

        # Progressive streaming - call s3gen for each chunk
        chunk_idx = 0
        token_pos = 0
        first_audio_time = None

        while token_pos < len(tokens):
            # Get chunk size and diffusion steps from schedule
            if chunk_idx < len(self.progressive_chunks):
                chunk_size = self.progressive_chunks[chunk_idx]
                n_steps = self.progressive_steps[min(chunk_idx, len(self.progressive_steps) - 1)]
            else:
                chunk_size = self.steady_chunk_size
                n_steps = self.steady_diffusion_steps

            chunk_end = min(token_pos + chunk_size, len(tokens))
            chunk_tokens = tokens[token_pos:chunk_end]

            # Call s3gen via BLS for this chunk
            audio = self._call_s3gen(chunk_tokens, s3gen_cond_bytes, n_timesteps=n_steps)

            if len(audio) > 0:
                if first_audio_time is None:
                    first_audio_time = time.time() - start_time
                    self.logger.log_info(
                        f"[{request_id}] First audio: {len(chunk_tokens)} tokens, "
                        f"{n_steps} steps, {first_audio_time*1000:.0f}ms"
                    )

                tensor = pb_utils.Tensor("audio", audio)
                sender.send(pb_utils.InferenceResponse(output_tensors=[tensor]))

            token_pos = chunk_end
            chunk_idx += 1

        sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        total_time = time.time() - start_time
        self.logger.log_info(
            f"[{request_id}] Complete: {len(tokens)} tokens, {chunk_idx} chunks, "
            f"first_audio={first_audio_time*1000:.0f}ms, total={total_time:.2f}s"
        )

    def finalize(self):
        """Cleanup."""
        self.logger.log_info("Chatterbox BLS: Finalizing...")
        if hasattr(self, 'triton_client'):
            self.triton_client.close()
