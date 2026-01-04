"""
Chatterbox Turbo TTS - Triton Python Backend with TensorRT

Uses TensorRT engines for non-autoregressive models (like Parakeet ASR):
- embed_tokens: TensorRT (token embedding)
- speech_encoder: TensorRT (reference audio -> conditioning)
- conditional_decoder: TensorRT (speech tokens -> mel)
- language_model: ONNX Runtime or vLLM (autoregressive generation)

Mel spectrogram computed in Python (torch.stft), not in TensorRT.
"""
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import triton_python_backend_utils as pb_utils


class TensorRTEngine:
    """TensorRT engine wrapper (same pattern as Parakeet)."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Get I/O info
        self.input_names = []
        self.output_names = []
        self.bindings = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            self.bindings[name] = {"shape": shape, "dtype": dtype}

    def infer(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Run inference with named inputs."""
        import torch

        # Set input shapes and addresses
        for name, tensor in inputs.items():
            if name in self.input_names:
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate outputs
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.bindings[name]["dtype"]
            torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
            out = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
            outputs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return outputs


class ONNXEngine:
    """ONNX Runtime engine wrapper (fallback for unsupported TRT ops)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference."""
        # Convert torch tensors to numpy if needed
        np_inputs = {}
        for name, val in inputs.items():
            if hasattr(val, 'cpu'):
                np_inputs[name] = val.cpu().numpy()
            else:
                np_inputs[name] = val

        outputs = self.session.run(self.output_names, np_inputs)
        return dict(zip(self.output_names, outputs))


class TritonPythonModel:
    """Chatterbox TTS with TensorRT acceleration."""

    def initialize(self, args: dict):
        import torch

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))
        engine_dir = os.path.join(model_dir, "engines")
        assets_dir = os.path.join(os.path.dirname(model_dir), "..", "chatterbox_assets")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        self.logger.log_info(f"Chatterbox: Loading from {model_dir}")
        self.logger.log_info(f"Chatterbox: Engine dir {engine_dir}")

        # Load engines (TensorRT preferred, ONNX fallback)
        self.engines = {}
        self._load_engine("embed_tokens", engine_dir)
        self._load_engine("speech_encoder", engine_dir)
        self._load_engine("conditional_decoder", engine_dir)
        self._load_engine("language_model", engine_dir, prefer_onnx=True)

        # Load tokenizer
        self._load_tokenizer(assets_dir)

        # Generation parameters
        self.temperature = 0.8
        self.top_k = 50
        self.max_tokens = 1024
        self.eos_token = 8193
        self.sample_rate = 24000

        # Decoupled mode for streaming
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.logger.log_info(f"Chatterbox: Initialized (decoupled={self.decoupled})")

    def _load_engine(self, name: str, engine_dir: str, prefer_onnx: bool = False):
        """Load TensorRT engine or ONNX fallback."""
        trt_path = os.path.join(engine_dir, f"{name}.engine")
        onnx_path = os.path.join(engine_dir, f"{name}_fp16.onnx")
        onnx_path_alt = os.path.join(engine_dir, f"{name}.onnx")

        if not prefer_onnx and os.path.exists(trt_path):
            try:
                self.engines[name] = TensorRTEngine(trt_path)
                self.logger.log_info(f"  {name}: TensorRT")
                return
            except Exception as e:
                self.logger.log_warn(f"  {name}: TensorRT failed ({e}), trying ONNX")

        # ONNX fallback
        for path in [onnx_path, onnx_path_alt]:
            if os.path.exists(path):
                try:
                    self.engines[name] = ONNXEngine(path)
                    self.logger.log_info(f"  {name}: ONNX Runtime")
                    return
                except Exception as e:
                    self.logger.log_warn(f"  {name}: ONNX failed ({e})")

        self.logger.log_warn(f"  {name}: NOT LOADED")

    def _load_tokenizer(self, assets_dir: str):
        """Load tokenizer from assets."""
        tokenizer_path = os.path.join(assets_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.logger.log_info("  tokenizer: Loaded")
            except Exception as e:
                self.logger.log_warn(f"  tokenizer: Failed ({e})")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def execute(self, requests: List) -> List:
        """Process TTS requests."""
        import torch

        responses = []

        for request in requests:
            request_id = request.request_id() or "unknown"
            start_time = time.time()

            try:
                # Get inputs
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = text_tensor.as_numpy()[0][0].decode("utf-8")

                ref_tensor = pb_utils.get_input_tensor_by_name(request, "reference_audio")

                self.logger.log_info(f"[{request_id}] Synthesizing: '{text[:40]}...'")

                # Process reference audio -> speaker embeddings and features
                if ref_tensor is not None:
                    ref_audio = ref_tensor.as_numpy().flatten()
                    speaker_embeddings, speaker_features = self._encode_speaker(ref_audio)
                else:
                    speaker_embeddings = torch.zeros(1, 256, device=self.device, dtype=torch.float16)
                    speaker_features = torch.zeros(1, 768, device=self.device, dtype=torch.float16)

                # Tokenize text
                text_tokens = self._tokenize(text)

                # Embed text tokens
                text_embeds = self._embed_tokens(text_tokens)

                # Generate speech tokens (autoregressive)
                if self.decoupled:
                    sender = request.get_response_sender()
                    self._generate_streaming(
                        sender, request_id, text_embeds,
                        speaker_embeddings, speaker_features, start_time
                    )
                else:
                    audio = self._generate_full(text_embeds, speaker_embeddings, speaker_features)
                    tensor = pb_utils.Tensor("audio", audio.astype(np.float32))
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

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize text."""
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            return np.array(encoded.ids, dtype=np.int64)
        else:
            # Fallback: character-level
            return np.array([ord(c) for c in text], dtype=np.int64)

    def _embed_tokens(self, token_ids: np.ndarray) -> "torch.Tensor":
        """Get embeddings for tokens using TensorRT/ONNX."""
        import torch

        if "embed_tokens" not in self.engines:
            raise RuntimeError("embed_tokens engine not loaded")

        token_tensor = torch.from_numpy(token_ids).unsqueeze(0).to(self.device)

        if isinstance(self.engines["embed_tokens"], TensorRTEngine):
            result = self.engines["embed_tokens"].infer({"input_ids": token_tensor})
            return result.get("embeddings", list(result.values())[0])
        else:
            result = self.engines["embed_tokens"].infer({"input_ids": token_ids.reshape(1, -1)})
            return torch.from_numpy(result.get("embeddings", list(result.values())[0])).to(self.device)

    def _encode_speaker(self, audio: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Encode reference audio to speaker embeddings and features."""
        import torch

        if "speech_encoder" not in self.engines:
            # Return placeholder embeddings and features
            embeddings = torch.zeros(1, 256, device=self.device, dtype=torch.float16)
            features = torch.zeros(1, 768, device=self.device, dtype=torch.float16)
            return embeddings, features

        # Compute mel spectrogram in Python (like Parakeet)
        mel = self._compute_mel(audio)  # [1, 80, time]

        if isinstance(self.engines["speech_encoder"], TensorRTEngine):
            result = self.engines["speech_encoder"].infer({"input_features": mel})
            embeddings = result.get("speaker_embeddings", list(result.values())[0])
            features = result.get("speaker_features", result.get("conditioning", embeddings))
        else:
            result = self.engines["speech_encoder"].infer({"input_features": mel.cpu().numpy()})
            embeddings = torch.from_numpy(
                result.get("speaker_embeddings", list(result.values())[0])
            ).to(self.device)
            features = torch.from_numpy(
                result.get("speaker_features", result.get("conditioning", embeddings.cpu().numpy()))
            ).to(self.device)

        return embeddings, features

    def _compute_mel(self, audio: np.ndarray) -> "torch.Tensor":
        """Compute mel spectrogram (like Parakeet does for STFT)."""
        import torch
        import torch.nn.functional as F

        # Convert to tensor
        audio = torch.from_numpy(audio.astype(np.float32)).to(self.device)

        # Normalize
        if audio.abs().max() > 1.0:
            audio = audio / 32768.0

        # Mel parameters (Chatterbox uses 80 mels at 24kHz)
        n_fft = 1024
        hop_length = 256
        n_mels = 80
        sample_rate = 24000

        # Resample if needed (assuming 16kHz input)
        if len(audio) > 0:
            # Simple upsampling from 16kHz to 24kHz
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                scale_factor=24000/16000,
                mode='linear',
                align_corners=False
            ).squeeze()

        # STFT
        window = torch.hann_window(n_fft, device=self.device)
        pad_amount = n_fft // 2
        audio_padded = F.pad(audio, (pad_amount, pad_amount), mode='reflect')

        stft = torch.stft(audio_padded, n_fft, hop_length, window=window, return_complex=True)
        magnitudes = stft.abs() ** 2

        # Mel filterbank
        mel_basis = self._mel_filterbank(sample_rate, n_fft, n_mels).to(self.device)
        mel_spec = torch.matmul(mel_basis, magnitudes)

        # Log mel
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        # Shape: [n_mels, time] -> [1, n_mels, time]
        return mel_spec.unsqueeze(0).to(torch.float16)

    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int) -> "torch.Tensor":
        """Create mel filterbank matrix."""
        import torch

        f_min, f_max = 0.0, sr / 2.0

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        freqs = mel_to_hz(mels)

        fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        filterbank = np.zeros((n_mels, len(fft_freqs)))
        for i in range(n_mels):
            low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
            for j, f in enumerate(fft_freqs):
                if low <= f <= center:
                    filterbank[i, j] = (f - low) / (center - low)
                elif center < f <= high:
                    filterbank[i, j] = (high - f) / (high - center)

        return torch.from_numpy(filterbank.astype(np.float32))

    def _lm_forward(self, inputs_embeds: "torch.Tensor") -> "torch.Tensor":
        """Run language model forward pass."""
        import torch

        if "language_model" not in self.engines:
            raise RuntimeError("language_model engine not loaded")

        batch_size, seq_len, _ = inputs_embeds.shape
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=self.device)
        position_ids = torch.arange(seq_len, dtype=torch.int64, device=self.device).unsqueeze(0)

        if isinstance(self.engines["language_model"], TensorRTEngine):
            result = self.engines["language_model"].infer({
                "inputs_embeds": inputs_embeds.to(torch.float16),
                "attention_mask": attention_mask,
                "position_ids": position_ids
            })
        else:
            result = self.engines["language_model"].infer({
                "inputs_embeds": inputs_embeds.cpu().numpy().astype(np.float32),
                "attention_mask": attention_mask.cpu().numpy(),
                "position_ids": position_ids.cpu().numpy()
            })

        logits = result.get("logits", list(result.values())[0])
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).to(self.device)
        return logits

    def _decode_to_audio(
        self,
        speech_tokens: np.ndarray,
        speaker_embeddings: "torch.Tensor",
        speaker_features: "torch.Tensor"
    ) -> np.ndarray:
        """Decode speech tokens to audio waveform using conditional decoder (vocoder)."""
        import torch

        if "conditional_decoder" not in self.engines:
            # Return silence with approximate duration
            return np.zeros(len(speech_tokens) * 256, dtype=np.float32)

        # Prepare inputs
        tokens = speech_tokens.reshape(1, -1).astype(np.int64)

        if isinstance(self.engines["conditional_decoder"], TensorRTEngine):
            token_tensor = torch.from_numpy(tokens).to(self.device)
            result = self.engines["conditional_decoder"].infer({
                "speech_tokens": token_tensor,
                "speaker_embeddings": speaker_embeddings.to(torch.float16),
                "speaker_features": speaker_features.to(torch.float16)
            })
            audio = result.get("waveform", list(result.values())[0])
            return audio.squeeze().cpu().numpy()
        else:
            result = self.engines["conditional_decoder"].infer({
                "speech_tokens": tokens,
                "speaker_embeddings": speaker_embeddings.cpu().numpy(),
                "speaker_features": speaker_features.cpu().numpy()
            })
            audio = result.get("waveform", list(result.values())[0])
            return audio.squeeze().astype(np.float32)

    def _sample_token(self, logits: "torch.Tensor", prev_tokens: List[int]) -> int:
        """Sample next token with temperature and top-k."""
        import torch

        logits = logits.float()

        # Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _generate_streaming(
        self,
        sender,
        request_id: str,
        text_embeds: "torch.Tensor",
        speaker_embeddings: "torch.Tensor",
        speaker_features: "torch.Tensor",
        start_time: float
    ):
        """Generate speech tokens and stream audio chunks."""
        import torch

        generated_tokens = []
        chunk_tokens = []
        chunk_size = 50  # ~2 seconds of audio per chunk

        for step in range(self.max_tokens):
            # Build input sequence
            if generated_tokens:
                speech_embeds = self._embed_tokens(np.array(generated_tokens, dtype=np.int64))
                inputs_embeds = torch.cat([text_embeds, speech_embeds], dim=1)
            else:
                inputs_embeds = text_embeds

            # Forward pass
            logits = self._lm_forward(inputs_embeds)

            # Sample next token
            next_token = self._sample_token(logits[0, -1], generated_tokens)
            generated_tokens.append(next_token)
            chunk_tokens.append(next_token)

            # Check EOS
            if next_token == self.eos_token:
                break

            # Stream chunk when we have enough tokens
            if len(chunk_tokens) >= chunk_size:
                audio = self._decode_to_audio(
                    np.array(chunk_tokens, dtype=np.int64),
                    speaker_embeddings,
                    speaker_features
                )
                tensor = pb_utils.Tensor("audio", audio.astype(np.float32))
                sender.send(pb_utils.InferenceResponse(output_tensors=[tensor]))
                chunk_tokens = []

        # Final chunk
        if chunk_tokens:
            audio = self._decode_to_audio(
                np.array(chunk_tokens, dtype=np.int64),
                speaker_embeddings,
                speaker_features
            )
            tensor = pb_utils.Tensor("audio", audio.astype(np.float32))
            sender.send(pb_utils.InferenceResponse(output_tensors=[tensor]))

        # Complete
        sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        elapsed = time.time() - start_time
        self.logger.log_info(
            f"[{request_id}] Complete: {len(generated_tokens)} tokens in {elapsed:.2f}s"
        )

    def _generate_full(
        self,
        text_embeds: "torch.Tensor",
        speaker_embeddings: "torch.Tensor",
        speaker_features: "torch.Tensor"
    ) -> np.ndarray:
        """Generate full audio without streaming."""
        import torch

        generated_tokens = []

        for step in range(self.max_tokens):
            if generated_tokens:
                speech_embeds = self._embed_tokens(np.array(generated_tokens, dtype=np.int64))
                inputs_embeds = torch.cat([text_embeds, speech_embeds], dim=1)
            else:
                inputs_embeds = text_embeds

            logits = self._lm_forward(inputs_embeds)
            next_token = self._sample_token(logits[0, -1], generated_tokens)
            generated_tokens.append(next_token)

            if next_token == self.eos_token:
                break

        # Decode all tokens to audio
        return self._decode_to_audio(
            np.array(generated_tokens, dtype=np.int64),
            speaker_embeddings,
            speaker_features
        )

    def finalize(self):
        self.logger.log_info("Chatterbox: Finalized")
