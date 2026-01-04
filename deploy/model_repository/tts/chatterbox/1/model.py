"""
Chatterbox TTS - Triton Python Backend

Architecture (Pure PyTorch with torch.compile):
- T3 (Triton vLLM backend): Text -> speech tokens
- S3Gen (PyTorch + torch.compile): Speech tokens -> audio waveform
  - Flow decoder: speech tokens -> mel spectrogram
  - HiFT vocoder: mel -> audio

This backend:
1. Loads pre-compiled voice conditioning for T3
2. Calls T3 via Triton gRPC for speech token generation
3. Runs S3Gen (PyTorch with torch.compile) for audio synthesis
4. Streams audio chunks progressively

Progressive Streaming Schedule:
- Chunks: [4, 8, 16, 32, 32, 32, 32, 32] tokens
- Steps:  [2, 3, 5, 7, 7, 7, 7, 7] diffusion steps
- Early chunks use fewer steps for lower latency
- Later chunks use more steps for better quality
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils

# Add chatterbox_tts to path
MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))


class TritonPythonModel:
    """Chatterbox TTS with T3 via Triton vLLM backend."""

    def initialize(self, args: dict):
        import torch
        import tritonclient.grpc as grpcclient
        from safetensors.torch import load_file

        self.model_config = json.loads(args["model_config"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        # Import chatterbox modules
        from chatterbox_tts import S3Gen, S3GEN_SR, S3_SR, SPEECH_VOCAB_SIZE, drop_invalid_tokens
        from chatterbox_tts import VoiceEncoder, T3Config
        from chatterbox_tts import punc_norm

        # Paths
        assets_dir = MODEL_DIR.parent / "chatterbox_assets"
        voices_dir = Path("/models/t3_weights/voices")  # Container path

        self.logger.log_info("Chatterbox TTS: Initializing (Pure PyTorch)")
        self.logger.log_info(f"  Assets: {assets_dir}")
        self.logger.log_info(f"  Voices: {voices_dir}")

        # Store constants
        self.SPEECH_VOCAB_SIZE = SPEECH_VOCAB_SIZE
        self.S3GEN_SR = S3GEN_SR
        self.S3_SR = S3_SR
        self.drop_invalid_tokens = drop_invalid_tokens
        self.punc_norm = punc_norm

        # T3 configuration (for token parsing)
        self.t3_config = T3Config()
        self.SPEECH_TOKEN_OFFSET = 2500  # Same as in T3 model

        # Load S3Gen (flow decoder + HiFT vocoder)
        self.logger.log_info("  Loading S3Gen...")
        s3gen_weights = assets_dir / "s3gen.safetensors"
        self.s3gen = S3Gen(use_fp16=True)
        if s3gen_weights.exists():
            self.s3gen.load_state_dict(load_file(str(s3gen_weights)), strict=False)
        self.s3gen = self.s3gen.to(self.device).eval()

        # Apply torch.compile for optimization
        use_compile = os.environ.get("USE_TORCH_COMPILE", "true").lower() == "true"
        if use_compile and hasattr(torch, 'compile'):
            self.logger.log_info("  Applying torch.compile() to S3Gen...")
            try:
                # Compile the inference method for best performance
                self.s3gen = torch.compile(self.s3gen, mode="reduce-overhead")
                self.logger.log_info("  torch.compile() applied successfully")
            except Exception as e:
                self.logger.log_warn(f"  torch.compile() failed: {e}")
                self.logger.log_warn("  Falling back to eager mode")
        else:
            self.logger.log_info("  Running in eager mode (torch.compile disabled)")

        # Load voice encoder
        self.logger.log_info("  Loading VoiceEncoder...")
        ve_weights = assets_dir / "ve.safetensors"
        self.voice_encoder = VoiceEncoder()
        if ve_weights.exists():
            self.voice_encoder.load_state_dict(load_file(str(ve_weights)))
        self.voice_encoder = self.voice_encoder.to(self.device).eval()

        # Load default S3Gen conditionals
        self.logger.log_info("  Loading default S3Gen conditionals...")
        self.default_s3gen_ref = None

        conds_path = assets_dir / "conds.pt"
        if conds_path.exists():
            conds_data = torch.load(conds_path, weights_only=True)
            # Use the 'gen' dict for S3Gen flow decoder
            self.default_s3gen_ref = conds_data.get('gen', conds_data)
            if isinstance(self.default_s3gen_ref, dict):
                for k, v in self.default_s3gen_ref.items():
                    if torch.is_tensor(v):
                        self.default_s3gen_ref[k] = v.to(self.device)
            self.logger.log_info("  Loaded default conditioning from conds.pt")
        else:
            self.logger.log_warn("  No default conditionals found")

        # Load compiled voice conditionings
        self.logger.log_info("  Loading compiled voices...")
        self.voices = {}  # T3 conditioning
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

                    # Handle different conditioning formats
                    if isinstance(voice_data, dict):
                        if 't3' in voice_data:
                            self.voices[voice_name] = voice_data['t3'].to(self.device)
                        self.logger.log_info(f"    Loaded voice: {voice_name} (dict format)")
                    else:
                        # Legacy format: just T3 conditioning tensor
                        self.voices[voice_name] = voice_data.to(self.device)
                        self.logger.log_info(f"    Loaded voice: {voice_name} ({voice_data.shape})")

        # Fallback: load conditioning.pt directly
        if not self.voices:
            compat_path = voices_dir.parent / "conditioning.pt"
            if compat_path.exists():
                conditioning = torch.load(compat_path, weights_only=True)
                self.voices["default"] = conditioning.to(self.device)
                self.logger.log_info(f"    Loaded fallback: default ({conditioning.shape})")

        self.logger.log_info(f"  Available voices: {list(self.voices.keys())}")
        self.logger.log_info(f"  Default voice: {self.default_voice}")

        # Triton client for T3
        triton_url = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
        self.logger.log_info(f"  T3 Triton URL: {triton_url}")
        self.t3_model_name = os.environ.get("T3_MODEL_NAME", "t3")
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)

        # Generation parameters
        self.temperature = float(os.environ.get("TTS_TEMPERATURE", "0.8"))
        self.top_p = float(os.environ.get("TTS_TOP_P", "0.8"))

        # Progressive streaming settings
        # Schedule: fewer steps for early chunks (low latency), more for later (quality)
        chunk_str = os.environ.get("TTS_PROGRESSIVE_CHUNKS", "4,8,16,32,32,32,32,32")
        self.progressive_chunks = [int(x) for x in chunk_str.split(",")]
        steps_str = os.environ.get("TTS_PROGRESSIVE_STEPS", "2,3,5,7,7,7,7,7")
        self.progressive_steps = [int(x) for x in steps_str.split(",")]
        self.steady_chunk_size = int(os.environ.get("TTS_CHUNK_TOKENS", "32"))
        self.steady_diffusion_steps = int(os.environ.get("S3GEN_DIFFUSION_STEPS", "7"))

        # Decoupled mode for streaming
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.logger.log_info(f"Chatterbox TTS: Initialized (decoupled={self.decoupled})")
        self.logger.log_info(f"  Progressive chunks: {self.progressive_chunks}")
        self.logger.log_info(f"  Progressive steps:  {self.progressive_steps}")

    def _get_voice_conditioning(self, voice_id: Optional[str] = None) -> "torch.Tensor":
        """Get conditioning tensor for the specified voice.

        Args:
            voice_id: Voice name (e.g., "default", "john", "sarah")
                      If None, uses the default voice

        Returns:
            Conditioning tensor [34, 1024] for T3
        """
        if voice_id is None or voice_id not in self.voices:
            voice_id = self.default_voice

        if voice_id not in self.voices:
            if self.voices:
                voice_id = list(self.voices.keys())[0]
            else:
                raise RuntimeError("No voices loaded")

        return self.voices[voice_id]

    def _prepare_s3gen_conditioning(self, ref_audio: Optional[np.ndarray] = None) -> dict:
        """Prepare S3Gen conditioning for flow decoder.

        Args:
            ref_audio: Optional reference audio for voice cloning

        Returns:
            S3Gen ref_dict with prompt_token, prompt_feat, embedding
        """
        if ref_audio is not None:
            # Compute S3Gen reference embeddings from reference audio
            import librosa

            ref_16k = ref_audio.astype(np.float32)

            # Get S3Gen reference embeddings for voice cloning
            s3gen_ref_wav = librosa.resample(ref_16k, orig_sr=16000, target_sr=self.S3GEN_SR)
            s3gen_ref = self.s3gen.embed_ref(s3gen_ref_wav, self.S3GEN_SR)

            return s3gen_ref
        else:
            # Use default conditioning
            if self.default_s3gen_ref is None:
                raise RuntimeError("No S3Gen conditioning available - load conds.pt")

            return self.default_s3gen_ref

    def _call_t3(self, text: str, conditioning: "torch.Tensor", stream: bool = False) -> List[int]:
        """Call T3 via Triton gRPC (vLLM backend).

        Args:
            text: Input text with [START] and [STOP] tokens
            conditioning: Voice conditioning tensor [34, 1024]
            stream: Whether to stream (not yet supported with vLLM backend)

        Returns:
            List of speech token IDs
        """
        import base64
        import io
        import torch
        import tritonclient.grpc as grpcclient

        # Prepare text input (vLLM backend expects "text_input")
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([text.encode()], dtype=object))

        # Serialize conditioning tensor as base64 for vLLM multimodal "image" input
        buffer = io.BytesIO()
        torch.save(conditioning.cpu(), buffer)
        cond_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        image_input = grpcclient.InferInput("image", [1], "BYTES")
        image_input.set_data_from_numpy(np.array([cond_b64.encode()], dtype=object))

        # Sampling parameters for vLLM backend
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

        # Request output
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
        # vLLM backend returns text, we need to extract token IDs
        # This depends on how the output is formatted
        text = output[0].decode() if isinstance(output[0], bytes) else str(output[0])

        # For now, assume output contains space-separated token IDs
        # TODO: Adjust based on actual vLLM backend output format
        tokens = []
        for part in text.split():
            try:
                token_id = int(part) - self.SPEECH_TOKEN_OFFSET
                if 0 <= token_id < self.SPEECH_VOCAB_SIZE:
                    tokens.append(token_id)
            except ValueError:
                pass
        return tokens

    def _decode_tokens_to_audio(
        self,
        speech_tokens,
        s3gen_cond: dict,
        n_timesteps: int = 7,
    ) -> np.ndarray:
        """Convert speech tokens to audio using S3Gen (PyTorch).

        Args:
            speech_tokens: Speech token IDs
            s3gen_cond: S3Gen ref_dict with conditioning
            n_timesteps: Diffusion steps for flow decoder
        """
        import torch

        if isinstance(speech_tokens, list):
            speech_tokens = torch.tensor(speech_tokens, device=self.device)

        # Clean tokens
        speech_tokens = self.drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < self.SPEECH_VOCAB_SIZE]

        if len(speech_tokens) == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            # S3Gen inference: flow decoder + HiFT vocoder
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=s3gen_cond,
                n_timesteps=n_timesteps,
            )
            return wav.squeeze().cpu().numpy().astype(np.float32)

    def execute(self, requests: List) -> List:
        """Process TTS requests."""
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

                # Get voice_id (optional)
                voice_tensor = pb_utils.get_input_tensor_by_name(request, "voice_id")
                voice_id = None
                if voice_tensor is not None:
                    voice_id = voice_tensor.as_numpy()[0].decode("utf-8")

                self.logger.log_info(f"[{request_id}] Text: '{text[:50]}...' voice={voice_id or self.default_voice}")

                # Get voice conditioning for T3
                t3_conditioning = self._get_voice_conditioning(voice_id)

                # Prepare S3Gen conditioning for flow decoder
                s3gen_cond = self._prepare_s3gen_conditioning(ref_audio)

                # Prepare text prompt
                text_prompt = "[START]" + self.punc_norm(text) + "[STOP]"

                if self.decoupled:
                    sender = request.get_response_sender()
                    self._generate_streaming(
                        sender, request_id, text_prompt,
                        t3_conditioning, s3gen_cond, start_time
                    )
                else:
                    # Non-streaming: call T3, decode all, return
                    tokens = self._call_t3(text_prompt, t3_conditioning)
                    audio = self._decode_tokens_to_audio(tokens, s3gen_cond)
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
        s3gen_cond: dict,
        start_time: float
    ):
        """Generate with progressive streaming."""
        import torch

        # Call T3 for all tokens (vLLM handles batching efficiently)
        tokens = self._call_t3(text_prompt, t3_conditioning)

        token_gen_time = time.time() - start_time
        self.logger.log_info(
            f"[{request_id}] Generated {len(tokens)} tokens in {token_gen_time:.2f}s"
        )

        # Progressive streaming
        chunk_idx = 0
        token_pos = 0
        first_audio_time = None

        while token_pos < len(tokens):
            # Get chunk size and diffusion steps
            if chunk_idx < len(self.progressive_chunks):
                chunk_size = self.progressive_chunks[chunk_idx]
                n_steps = self.progressive_steps[min(chunk_idx, len(self.progressive_steps) - 1)]
            else:
                chunk_size = self.steady_chunk_size
                n_steps = self.steady_diffusion_steps

            chunk_end = min(token_pos + chunk_size, len(tokens))
            chunk_tokens = tokens[token_pos:chunk_end]

            # Decode chunk
            audio = self._decode_tokens_to_audio(chunk_tokens, s3gen_cond, n_timesteps=n_steps)

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
        self.logger.log_info("Chatterbox: Finalizing...")
        if hasattr(self, 'triton_client'):
            self.triton_client.close()
        import torch
        torch.cuda.empty_cache()
