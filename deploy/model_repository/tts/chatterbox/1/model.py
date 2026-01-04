"""
Chatterbox Turbo TTS - Triton Python Backend

Architecture:
- T3 (Triton vLLM backend): Text -> speech tokens (uses pre-loaded conditioning)
- Vocoder: Speech tokens -> audio waveform
  - Option 1: TensorRT (conditional_decoder.engine) - fastest
  - Option 2: S3Gen (safetensors) - PyTorch fallback

This backend:
1. Loads pre-compiled voice conditioning for T3 and vocoder
2. Calls T3 via Triton gRPC (vLLM backend with pre-loaded conditioning)
3. Decodes speech tokens to audio using TRT vocoder (or S3Gen fallback)
4. Streams audio chunks progressively

Progressive Streaming for Low Latency:
- T3 generates all tokens first (vLLM handles efficiently)
- Audio decoded in chunks: 4, 8, 16, 32, 32... tokens
- Diffusion steps per chunk: 1, 2, 5, 7, 10... (fewer = faster, S3Gen only)
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import triton_python_backend_utils as pb_utils

# Add chatterbox_tts to path
MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))


class TRTVocoder:
    """TensorRT engine wrapper for conditional_decoder vocoder."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Get binding info
        self.input_names = []
        self.output_names = []
        self.bindings = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            self.bindings[name] = {"dtype": dtype}

    def __call__(self, speech_tokens, speaker_embeddings, speaker_features):
        """Run TRT vocoder inference."""
        import torch

        # Ensure inputs are on GPU as contiguous tensors
        if not torch.is_tensor(speech_tokens):
            speech_tokens = torch.from_numpy(speech_tokens)
        if not torch.is_tensor(speaker_embeddings):
            speaker_embeddings = torch.from_numpy(speaker_embeddings)
        if not torch.is_tensor(speaker_features):
            speaker_features = torch.from_numpy(speaker_features)

        speech_tokens = speech_tokens.cuda().contiguous().to(torch.int64)
        speaker_embeddings = speaker_embeddings.cuda().contiguous().to(torch.float32)
        speaker_features = speaker_features.cuda().contiguous().to(torch.float32)

        # Set input shapes and addresses
        self.context.set_input_shape("speech_tokens", tuple(speech_tokens.shape))
        self.context.set_tensor_address("speech_tokens", speech_tokens.data_ptr())
        self.context.set_tensor_address("speaker_embeddings", speaker_embeddings.data_ptr())
        self.context.set_tensor_address("speaker_features", speaker_features.data_ptr())

        # Allocate output
        out_shape = self.context.get_tensor_shape(self.output_names[0])
        output = torch.empty(tuple(out_shape), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.output_names[0], output.data_ptr())

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return output.cpu().numpy()


class TRTFlowDecoder:
    """TensorRT engine wrapper for unrolled flow matching decoder.

    This engine has N diffusion steps baked in (unrolled), enabling
    single-call inference instead of iterating in Python.
    """

    def __init__(self, engine_path: str, config_path: str = None):
        import tensorrt as trt
        import torch

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Load config if available
        self.num_steps = 4  # Default
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)
                self.num_steps = config.get('num_steps', 4)

        # Get binding info
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def __call__(
        self,
        latents: "torch.Tensor",
        speaker_embedding: "torch.Tensor",
        speech_token_embedding: "torch.Tensor",
    ) -> "torch.Tensor":
        """Run unrolled flow decoder inference.

        Args:
            latents: Initial noise [batch, mel_channels, time_frames]
            speaker_embedding: Speaker conditioning [batch, speaker_dim]
            speech_token_embedding: Token features [batch, seq_len, hidden_dim]

        Returns:
            Denoised mel spectrogram [batch, mel_channels, time_frames]
        """
        import torch

        # Ensure inputs are on GPU as contiguous tensors
        latents = latents.cuda().contiguous().to(torch.float16)
        speaker_embedding = speaker_embedding.cuda().contiguous().to(torch.float16)
        speech_token_embedding = speech_token_embedding.cuda().contiguous().to(torch.float16)

        # Set input shapes and addresses
        self.context.set_input_shape("latents", tuple(latents.shape))
        self.context.set_tensor_address("latents", latents.data_ptr())
        self.context.set_tensor_address("speaker_embedding", speaker_embedding.data_ptr())
        self.context.set_tensor_address("speech_token_embedding", speech_token_embedding.data_ptr())

        # Get output shape and allocate
        out_shape = self.context.get_tensor_shape(self.output_names[0])
        output = torch.empty(tuple(out_shape), dtype=torch.float16, device="cuda")
        self.context.set_tensor_address(self.output_names[0], output.data_ptr())

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return output.float()  # Return as float32 for downstream processing


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
        engines_dir = MODEL_DIR / "engines"

        self.logger.log_info("Chatterbox Turbo: Initializing")
        self.logger.log_info(f"  Assets: {assets_dir}")
        self.logger.log_info(f"  Voices: {voices_dir}")
        self.logger.log_info(f"  Engines: {engines_dir}")

        # Store constants
        self.SPEECH_VOCAB_SIZE = SPEECH_VOCAB_SIZE
        self.S3GEN_SR = S3GEN_SR
        self.S3_SR = S3_SR
        self.drop_invalid_tokens = drop_invalid_tokens
        self.punc_norm = punc_norm

        # T3 configuration (for token parsing)
        self.t3_config = T3Config()
        self.SPEECH_TOKEN_OFFSET = 2500  # Same as in T3 model

        # Vocoder loading priority: TRT > S3Gen (safetensors)
        self.vocoder_type = None  # 'trt' or 's3gen'
        self.vocoder = None

        # Try TensorRT first (fastest)
        trt_vocoder_path = engines_dir / "conditional_decoder.engine"
        if trt_vocoder_path.exists():
            try:
                self.logger.log_info(f"  Loading TensorRT vocoder: {trt_vocoder_path.name}")
                self.vocoder = TRTVocoder(str(trt_vocoder_path))
                self.vocoder_type = 'trt'
                self.logger.log_info("  TensorRT vocoder loaded successfully")
            except Exception as e:
                self.logger.log_warn(f"  Failed to load TRT vocoder: {e}")

        # Fall back to S3Gen (PyTorch safetensors)
        if self.vocoder_type is None:
            self.logger.log_info("  Loading S3Gen vocoder (safetensors)...")
            s3gen_weights = assets_dir / "s3gen.safetensors"
            self.s3gen = S3Gen(use_fp16=True)
            if s3gen_weights.exists():
                self.s3gen.load_state_dict(load_file(str(s3gen_weights)), strict=False)
            self.s3gen = self.s3gen.to(self.device).eval()
            self.vocoder_type = 's3gen'

            # Load voice encoder (only needed for S3Gen path)
            self.logger.log_info("  Loading VoiceEncoder...")
            ve_weights = assets_dir / "ve.safetensors"
            self.voice_encoder = VoiceEncoder()
            if ve_weights.exists():
                self.voice_encoder.load_state_dict(load_file(str(ve_weights)))
            self.voice_encoder = self.voice_encoder.to(self.device).eval()

        self.logger.log_info(f"  Vocoder: {self.vocoder_type.upper()}")

        # Load TRT flow decoder if available (unrolled diffusion steps)
        self.flow_decoder_trt = None
        flow_config_path = engines_dir / "flow_decoder_config.json"
        if flow_config_path.exists():
            try:
                with open(flow_config_path) as f:
                    flow_config = json.load(f)
                engine_name = flow_config.get("engine_file", "flow_decoder_4steps.engine")
                flow_engine_path = engines_dir / engine_name
                if flow_engine_path.exists():
                    self.logger.log_info(f"  Loading TRT flow decoder: {engine_name}")
                    self.flow_decoder_trt = TRTFlowDecoder(
                        str(flow_engine_path),
                        str(flow_config_path)
                    )
                    self.logger.log_info(f"  Flow decoder TRT loaded ({self.flow_decoder_trt.num_steps} steps)")
            except Exception as e:
                self.logger.log_warn(f"  Failed to load TRT flow decoder: {e}")

        # Load default vocoder conditionals
        self.logger.log_info("  Loading default vocoder conditionals...")
        self.default_vocoder_cond = None
        self.default_s3gen_ref = None

        conds_path = assets_dir / "conds.pt"
        if conds_path.exists():
            conds_data = torch.load(conds_path, weights_only=True)
            if self.vocoder_type == 'trt':
                # For TRT vocoder, we need speaker_embeddings and speaker_features
                # These should be pre-computed by speech_encoder during build
                if 'vocoder' in conds_data:
                    self.default_vocoder_cond = conds_data['vocoder']
                    self.logger.log_info("  Loaded TRT vocoder conditioning from conds.pt")
                else:
                    self.logger.log_warn("  No TRT vocoder conditioning in conds.pt")
            else:
                # For S3Gen, use the 'gen' dict
                self.default_s3gen_ref = conds_data.get('gen', {})
                for k, v in self.default_s3gen_ref.items():
                    if torch.is_tensor(v):
                        self.default_s3gen_ref[k] = v.to(self.device)
        else:
            self.logger.log_warn("  No default conditionals found")

        # Load compiled voice conditionings
        self.logger.log_info("  Loading compiled voices...")
        self.voices = {}  # T3 conditioning
        self.vocoder_voices = {}  # TRT vocoder conditioning
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
                        # New format with both T3 and vocoder conditioning
                        if 't3' in voice_data:
                            self.voices[voice_name] = voice_data['t3'].to(self.device)
                        if 'vocoder' in voice_data and self.vocoder_type == 'trt':
                            self.vocoder_voices[voice_name] = voice_data['vocoder']
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
        if self.vocoder_type == 'trt':
            self.logger.log_info(f"  Vocoder voices: {list(self.vocoder_voices.keys())}")

        # Triton client for T3
        triton_url = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
        self.logger.log_info(f"  T3 Triton URL: {triton_url}")
        self.t3_model_name = os.environ.get("T3_MODEL_NAME", "t3")
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)

        # Generation parameters
        self.temperature = float(os.environ.get("TTS_TEMPERATURE", "0.8"))
        self.top_p = float(os.environ.get("TTS_TOP_P", "0.8"))

        # Progressive streaming settings
        chunk_str = os.environ.get("TTS_PROGRESSIVE_CHUNKS", "4,8,16,32,32")
        self.progressive_chunks = [int(x) for x in chunk_str.split(",")]
        steps_str = os.environ.get("TTS_PROGRESSIVE_STEPS", "1,2,5,7,10")
        self.progressive_steps = [int(x) for x in steps_str.split(",")]
        self.steady_chunk_size = int(os.environ.get("TTS_CHUNK_TOKENS", "32"))
        self.steady_diffusion_steps = int(os.environ.get("S3GEN_DIFFUSION_STEPS", "10"))

        # Decoupled mode for streaming
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.logger.log_info(f"Chatterbox Turbo: Initialized (decoupled={self.decoupled})")
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

    def _prepare_vocoder_conditioning(self, ref_audio: Optional[np.ndarray] = None, voice_id: Optional[str] = None) -> Union[dict, tuple]:
        """Prepare conditioning for vocoder.

        Args:
            ref_audio: Optional reference audio for voice cloning
            voice_id: Voice ID for pre-compiled conditioning

        Returns:
            Either:
                - dict (S3Gen): ref_dict with prompt_token, prompt_feat, embedding
                - tuple (TRT): (speaker_embeddings, speaker_features)
        """
        import torch

        if self.vocoder_type == 'trt':
            # TRT vocoder needs speaker_embeddings and speaker_features
            # These must be pre-computed (runtime computation not supported)
            if voice_id and voice_id in self.vocoder_voices:
                return self.vocoder_voices[voice_id]
            elif self.default_vocoder_cond is not None:
                return self.default_vocoder_cond
            else:
                raise RuntimeError("No TRT vocoder conditioning available. Run build with speech_encoder.")
        else:
            # S3Gen path
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
        vocoder_cond: Union[dict, tuple],
        n_timesteps: int = 10,
    ) -> np.ndarray:
        """Convert speech tokens to audio using vocoder.

        Args:
            speech_tokens: Speech token IDs
            vocoder_cond: Either:
                - dict (S3Gen): ref_dict with prompt_token, prompt_feat, embedding
                - tuple (TRT): (speaker_embeddings, speaker_features)
            n_timesteps: Diffusion steps (S3Gen only, ignored if TRT flow decoder)
        """
        import torch

        if isinstance(speech_tokens, list):
            speech_tokens = torch.tensor(speech_tokens, device=self.device)

        # Clean tokens
        speech_tokens = self.drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < self.SPEECH_VOCAB_SIZE]

        if len(speech_tokens) == 0:
            return np.zeros(0, dtype=np.float32)

        if self.vocoder_type == 'trt':
            # TRT vocoder path (conditional_decoder handles everything)
            speaker_embeddings, speaker_features = vocoder_cond
            wav = self.vocoder(
                speech_tokens.unsqueeze(0) if speech_tokens.dim() == 1 else speech_tokens,
                speaker_embeddings,
                speaker_features,
            )
            return wav.squeeze().astype(np.float32)

        elif self.flow_decoder_trt is not None:
            # TRT flow decoder + PyTorch vocoder path
            # Flow decoder has fixed steps baked in, so n_timesteps is ignored
            with torch.inference_mode():
                # Embed tokens
                token_embedding = self.s3gen.embed_tokens(speech_tokens.unsqueeze(0))

                # Get speaker conditioning
                ref_dict = vocoder_cond
                speaker_embedding = ref_dict.get('embedding', ref_dict.get('speaker_embedding'))
                if speaker_embedding is None:
                    raise RuntimeError("No speaker embedding in vocoder_cond for TRT flow decoder")

                # Generate initial noise
                batch_size = 1
                mel_channels = 128  # S3Gen default
                time_frames = token_embedding.shape[1] * 4  # Approximate mel length
                latents = torch.randn(batch_size, mel_channels, time_frames, device=self.device)

                # Run TRT flow decoder (all diffusion steps in single call)
                mel = self.flow_decoder_trt(
                    latents,
                    speaker_embedding,
                    token_embedding,
                )

                # Run PyTorch vocoder (HiFiGAN) on mel spectrogram
                wav = self.s3gen.vocoder(mel)

            return wav.squeeze().cpu().numpy().astype(np.float32)

        else:
            # Full PyTorch S3Gen path
            with torch.inference_mode():
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=vocoder_cond,
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

                # Prepare vocoder conditioning (S3Gen dict or TRT tuple)
                vocoder_cond = self._prepare_vocoder_conditioning(ref_audio, voice_id)

                # Prepare text prompt
                text_prompt = "[START]" + self.punc_norm(text) + "[STOP]"

                if self.decoupled:
                    sender = request.get_response_sender()
                    self._generate_streaming(
                        sender, request_id, text_prompt,
                        t3_conditioning, vocoder_cond, start_time
                    )
                else:
                    # Non-streaming: call T3, decode all, return
                    tokens = self._call_t3(text_prompt, t3_conditioning)
                    audio = self._decode_tokens_to_audio(tokens, vocoder_cond)
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
        vocoder_cond: Union[dict, tuple],
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
            audio = self._decode_tokens_to_audio(chunk_tokens, vocoder_cond, n_timesteps=n_steps)

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
