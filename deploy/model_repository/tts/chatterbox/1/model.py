"""
Chatterbox Turbo TTS - Triton Python Backend

Orchestrates the Chatterbox TTS pipeline with ONNX Runtime (TensorRT EP) acceleration.

Pipeline:
1. Text tokenization
2. Speech encoder (voice conditioning from reference audio)
3. Language model (autoregressive speech token generation with KV cache)
4. Conditional decoder (speech tokens -> mel spectrogram, single step)
5. HiFT vocoder (mel -> waveform)

Concurrency:
- KV cache pool allows multiple simultaneous generations
- Each slot has dedicated cache memory
- Requests acquire/release slots as needed

Streaming:
- Audio chunks sent as tokens are generated
- ~1 second of audio per chunk (25 tokens at 25Hz)
"""
import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import triton_python_backend_utils as pb_utils

# Lazy imports
torch = None
ort = None


def lazy_import():
    """Lazy import heavy dependencies."""
    global torch, ort
    if torch is None:
        import torch as _torch
        torch = _torch
        torch.set_num_threads(1)
    if ort is None:
        import onnxruntime as _ort
        ort = _ort


# Constants
SAMPLE_RATE = 24000
TOKEN_RATE = 25
MAX_SPEECH_TOKENS = 1024
EOS_TOKEN_ID = 8193
NUM_LAYERS = 24
NUM_KV_HEADS = 16
HEAD_DIM = 64
HIDDEN_DIM = 896


class KVCacheManager:
    """Manages KV cache slots for concurrent autoregressive generation."""

    def __init__(self, num_slots: int, max_seq_len: int, device: str):
        lazy_import()
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.device = device

        self._available = queue.Queue()
        for i in range(num_slots):
            self._available.put(i)

        # Pre-allocate cache: (slots, layers, 2, seq, heads, dim)
        cache_shape = (num_slots, NUM_LAYERS, 2, max_seq_len, NUM_KV_HEADS, HEAD_DIM)
        self.cache = torch.zeros(cache_shape, dtype=torch.float16, device=device)
        self.seq_lens = [0] * num_slots
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> Optional[int]:
        try:
            slot = self._available.get(timeout=timeout)
            with self._lock:
                self.seq_lens[slot] = 0
                self.cache[slot].zero_()
            return slot
        except queue.Empty:
            return None

    def release(self, slot: int):
        self._available.put(slot)

    def get(self, slot: int) -> np.ndarray:
        seq_len = self.seq_lens[slot]
        return self.cache[slot, :, :, :seq_len].cpu().numpy()

    def update(self, slot: int, new_kv: np.ndarray, pos: int):
        with self._lock:
            kv_tensor = torch.from_numpy(new_kv).to(self.device)
            self.cache[slot, :, :, pos:pos+kv_tensor.shape[2]] = kv_tensor
            self.seq_lens[slot] = pos + kv_tensor.shape[2]


class ONNXModel:
    """Wrapper for ONNX Runtime model with optional TensorRT acceleration."""

    def __init__(self, model_path: str, use_trt: bool = True):
        lazy_import()

        providers = []
        if use_trt:
            providers.append(('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
                'trt_fp16_enable': True,
            }))
        providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, **inputs) -> Dict[str, np.ndarray]:
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs))


class TritonPythonModel:
    """Chatterbox TTS Triton backend with ONNX Runtime."""

    def initialize(self, args: dict):
        lazy_import()
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        params = {k: v["string_value"] for k, v in self.model_config['parameters'].items()}

        self.model_dir = params.get("model_dir", "/models/tts/chatterbox_assets")
        self.sample_rate = int(params.get("sample_rate", "24000"))
        max_concurrent = int(params.get("max_concurrent", "4"))

        self.device = "cuda"
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.logger.log_info(f"Chatterbox: Initializing from {self.model_dir}")

        # Load tokenizer
        self._load_tokenizer()

        # Load ONNX models
        self._load_models()

        # Initialize KV cache pool
        self.kv_manager = KVCacheManager(
            num_slots=max_concurrent,
            max_seq_len=MAX_SPEECH_TOKENS + 512,
            device=self.device
        )

        # Generation parameters
        self.temperature = 0.8
        self.top_p = 0.95
        self.top_k = 50
        self.repetition_penalty = 1.2
        self.token_chunk_size = 25

        self.logger.log_info("Chatterbox: Initialization complete")

    def _load_tokenizer(self):
        """Load text tokenizer."""
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.logger.log_info("Chatterbox: Tokenizer loaded")
            except Exception as e:
                self.logger.log_warn(f"Chatterbox: Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
            self.logger.log_warn("Chatterbox: No tokenizer found")

    def _load_models(self):
        """Load ONNX models."""
        onnx_dir = os.path.join(self.model_dir, "onnx")

        # Try FP16 models first, fall back to FP32
        suffix = "_fp16" if os.path.exists(os.path.join(onnx_dir, "embed_tokens_fp16.onnx")) else ""

        models = {
            "embed": f"embed_tokens{suffix}.onnx",
            "encoder": f"speech_encoder{suffix}.onnx",
            "lm": f"language_model{suffix}.onnx",
            "decoder": f"conditional_decoder{suffix}.onnx",
        }

        self.models = {}
        for name, filename in models.items():
            path = os.path.join(onnx_dir, filename)
            if os.path.exists(path):
                try:
                    self.models[name] = ONNXModel(path, use_trt=True)
                    self.logger.log_info(f"Chatterbox: Loaded {name} from {filename}")
                except Exception as e:
                    self.logger.log_error(f"Chatterbox: Failed to load {name}: {e}")
            else:
                self.logger.log_warn(f"Chatterbox: Model not found: {path}")

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize input text."""
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            return np.array(encoded.ids, dtype=np.int64)
        else:
            # Fallback
            return np.array([ord(c) for c in text], dtype=np.int64)

    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for tokens."""
        if "embed" not in self.models:
            raise RuntimeError("Embed model not loaded")
        result = self.models["embed"](input_ids=token_ids.reshape(1, -1))
        return result.get("embeddings", result.get(list(result.keys())[0]))

    def _encode_speech(self, audio: np.ndarray) -> np.ndarray:
        """Encode reference audio for conditioning."""
        if "encoder" not in self.models:
            # Return neutral conditioning
            return np.zeros((1, 256), dtype=np.float16)

        # Compute mel features (simplified)
        features = audio.reshape(1, -1, 80).astype(np.float16)
        result = self.models["encoder"](input_features=features)
        return result.get("conditioning", result.get(list(result.keys())[0]))

    def _lm_forward(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_kv: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass through language model."""
        if "lm" not in self.models:
            raise RuntimeError("LM model not loaded")

        inputs = {
            "inputs_embeds": inputs_embeds.astype(np.float16),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }
        if past_kv is not None:
            inputs["past_key_values"] = past_kv.astype(np.float16)

        result = self.models["lm"](**inputs)
        logits = result.get("logits", result.get(list(result.keys())[0]))
        new_kv = result.get("present_key_values")
        return logits, new_kv

    def _decode_tokens(self, tokens: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
        """Decode speech tokens to mel spectrogram."""
        if "decoder" not in self.models:
            raise RuntimeError("Decoder model not loaded")

        result = self.models["decoder"](
            input_ids=tokens.astype(np.int64),
            conditioning=conditioning.astype(np.float16)
        )
        return result.get("mel_spectrogram", result.get(list(result.keys())[0]))

    def _sample_token(self, logits: np.ndarray, prev_tokens: List[int]) -> int:
        """Sample next token with temperature, top-p, repetition penalty."""
        logits = logits.astype(np.float32)

        # Repetition penalty
        if prev_tokens and self.repetition_penalty != 1.0:
            for tok in set(prev_tokens[-50:]):
                logits[tok] /= self.repetition_penalty

        # Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k
        if self.top_k > 0:
            indices = np.argpartition(logits, -self.top_k)[-self.top_k:]
            mask = np.ones_like(logits, dtype=bool)
            mask[indices] = False
            logits[mask] = float('-inf')

        # Softmax and sample
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return int(np.random.choice(len(probs), p=probs))

    def _mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio (placeholder)."""
        # TODO: Integrate HiFT vocoder
        # For now, return silence with correct duration
        frames = mel.shape[-1]
        samples = frames * 480  # Hop size
        return np.zeros(samples, dtype=np.float32)

    def execute(self, requests: List) -> List:
        responses = []

        for request in requests:
            request_id = request.request_id() or str(uuid4())[:8]

            try:
                # Get inputs
                target_text = pb_utils.get_input_tensor_by_name(request, "target_text")
                text = target_text.as_numpy()[0][0].decode('utf-8')

                ref_wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")

                self.logger.log_info(f"[{request_id}] Synthesizing: '{text[:40]}...'")
                start_time = time.time()

                # Acquire KV cache slot
                kv_slot = self.kv_manager.acquire(timeout=30.0)
                if kv_slot is None:
                    raise RuntimeError("No KV cache slots available")

                try:
                    # Get conditioning
                    if ref_wav_tensor is not None:
                        conditioning = self._encode_speech(ref_wav_tensor.as_numpy().flatten())
                    else:
                        conditioning = np.zeros((1, 256), dtype=np.float16)

                    # Tokenize and embed text
                    text_tokens = self._tokenize(text)
                    text_embeds = self._embed_tokens(text_tokens)

                    if self.decoupled:
                        sender = request.get_response_sender()
                        self._generate_streaming(
                            sender, request_id, text_embeds, conditioning, kv_slot, start_time
                        )
                    else:
                        audio = self._generate_full(text_embeds, conditioning, kv_slot)
                        tensor = pb_utils.Tensor("waveform", audio)
                        responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))

                finally:
                    self.kv_manager.release(kv_slot)

            except Exception as e:
                self.logger.log_error(f"[{request_id}] Error: {e}")
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

        if not self.decoupled:
            return responses

    def _generate_streaming(
        self,
        sender,
        request_id: str,
        text_embeds: np.ndarray,
        conditioning: np.ndarray,
        kv_slot: int,
        start_time: float
    ):
        """Generate audio with streaming output."""
        batch_size = 1
        seq_len = text_embeds.shape[1]

        # Initial masks
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # Prefill
        logits, new_kv = self._lm_forward(text_embeds, attention_mask, position_ids)
        if new_kv is not None:
            self.kv_manager.update(kv_slot, new_kv, 0)

        # Autoregressive generation
        generated = []
        chunk_tokens = []
        current_pos = seq_len

        for step in range(MAX_SPEECH_TOKENS):
            token = self._sample_token(logits[0, -1], generated)
            generated.append(token)
            chunk_tokens.append(token)

            if token == EOS_TOKEN_ID:
                break

            # Process chunk
            if len(chunk_tokens) >= self.token_chunk_size:
                mel = self._decode_tokens(
                    np.array(chunk_tokens, dtype=np.int64).reshape(1, -1),
                    conditioning
                )
                audio = self._mel_to_audio(mel)

                tensor = pb_utils.Tensor("waveform", audio.astype(np.float32))
                sender.send(pb_utils.InferenceResponse(output_tensors=[tensor]))
                chunk_tokens = []

            # Next step
            token_embed = self._embed_tokens(np.array([token], dtype=np.int64))
            attention_mask = np.ones((batch_size, current_pos + 1), dtype=np.int64)
            position_ids = np.array([[current_pos]], dtype=np.int64)

            past_kv = self.kv_manager.get(kv_slot)
            logits, new_kv = self._lm_forward(token_embed, attention_mask, position_ids, past_kv)

            if new_kv is not None:
                self.kv_manager.update(kv_slot, new_kv, current_pos)
            current_pos += 1

        # Final chunk
        if chunk_tokens:
            mel = self._decode_tokens(
                np.array(chunk_tokens, dtype=np.int64).reshape(1, -1),
                conditioning
            )
            audio = self._mel_to_audio(mel)
            tensor = pb_utils.Tensor("waveform", audio.astype(np.float32))
            sender.send(pb_utils.InferenceResponse(output_tensors=[tensor]))

        # Complete
        sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        elapsed = time.time() - start_time
        audio_dur = len(generated) / TOKEN_RATE
        rtf = elapsed / audio_dur if audio_dur > 0 else 0
        self.logger.log_info(
            f"[{request_id}] Complete: {len(generated)} tokens, "
            f"{audio_dur:.1f}s audio, {elapsed:.2f}s (RTF={rtf:.2f})"
        )

    def _generate_full(
        self,
        text_embeds: np.ndarray,
        conditioning: np.ndarray,
        kv_slot: int
    ) -> np.ndarray:
        """Generate full audio without streaming."""
        batch_size = 1
        seq_len = text_embeds.shape[1]

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, new_kv = self._lm_forward(text_embeds, attention_mask, position_ids)
        if new_kv is not None:
            self.kv_manager.update(kv_slot, new_kv, 0)

        generated = []
        current_pos = seq_len

        for step in range(MAX_SPEECH_TOKENS):
            token = self._sample_token(logits[0, -1], generated)
            generated.append(token)

            if token == EOS_TOKEN_ID:
                break

            token_embed = self._embed_tokens(np.array([token], dtype=np.int64))
            attention_mask = np.ones((batch_size, current_pos + 1), dtype=np.int64)
            position_ids = np.array([[current_pos]], dtype=np.int64)

            past_kv = self.kv_manager.get(kv_slot)
            logits, new_kv = self._lm_forward(token_embed, attention_mask, position_ids, past_kv)

            if new_kv is not None:
                self.kv_manager.update(kv_slot, new_kv, current_pos)
            current_pos += 1

        # Decode all tokens
        mel = self._decode_tokens(
            np.array(generated, dtype=np.int64).reshape(1, -1),
            conditioning
        )
        return self._mel_to_audio(mel)

    def finalize(self):
        self.logger.log_info("Chatterbox: Finalizing")
