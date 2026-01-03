"""
Parakeet TDT 0.6B V2 - Triton Python Backend

Uses TensorRT engines for GPU inference.
Architecture: encoder + decoder_joint (combined decoder/joiner from HuggingFace).
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TensorRTInference:
    """TensorRT engine wrapper with dynamic shape support."""

    def __init__(self, engine_path, device_id=0):
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
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            self.bindings[name] = {"shape": shape, "dtype": dtype}

    def infer(self, inputs_dict):
        """Run inference with named inputs."""
        import torch

        # Set input shapes and addresses
        for name, tensor in inputs_dict.items():
            if name in self.input_names:
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate and set outputs
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


class TritonPythonModel:
    """Parakeet TDT ASR using TensorRT GPU."""

    def initialize(self, args):
        """Load pre-built TensorRT engines for GPU inference."""
        import torch

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))
        engine_dir = os.path.join(model_dir, "engines")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pb_utils.Logger.log_info(f"Loading Parakeet from {model_dir}")

        # Check for TensorRT engines
        encoder_engine = os.path.join(engine_dir, "encoder.engine")
        decoder_joint_engine = os.path.join(engine_dir, "decoder_joint.engine")

        if not os.path.exists(encoder_engine):
            raise RuntimeError(f"Encoder engine not found: {encoder_engine}")
        if not os.path.exists(decoder_joint_engine):
            raise RuntimeError(f"Decoder_joint engine not found: {decoder_joint_engine}")

        pb_utils.Logger.log_info("Loading TensorRT engines...")
        self.encoder = TensorRTInference(encoder_engine)
        self.decoder_joint = TensorRTInference(decoder_joint_engine)
        pb_utils.Logger.log_info("TensorRT engines loaded")

        # Load vocabulary (try vocab.txt first, then tokens.txt)
        vocab_path = os.path.join(model_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(model_dir, "tokens.txt")
        self.vocab = self._load_vocab(vocab_path)
        self.blank_id = 0  # Usually blank is 0

        pb_utils.Logger.log_info(f"Parakeet initialized: vocab={len(self.vocab)}, device={self.device}")

    def _load_vocab(self, path):
        """Load token vocabulary."""
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    vocab[i] = token
        return vocab

    def execute(self, requests):
        """Run inference on audio."""
        import torch

        responses = []

        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio")
            audio = audio.as_numpy().flatten().astype(np.float32)

            # Normalize
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            with torch.no_grad():
                text = self._transcribe(audio)

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def _compute_features(self, audio):
        """Compute mel spectrogram features."""
        import torch
        import torch.nn.functional as F

        # Convert to tensor
        audio = torch.from_numpy(audio).to(self.device)

        # Parameters for mel spectrogram (Parakeet uses 128 mels)
        n_fft = 512
        hop_length = 160
        n_mels = 128
        sample_rate = 16000

        # Simple STFT-based mel spectrogram
        window = torch.hann_window(n_fft, device=self.device)

        # Pad audio
        pad_amount = n_fft // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')

        # STFT
        stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
        magnitudes = stft.abs() ** 2

        # Mel filterbank
        mel_basis = self._mel_filterbank(sample_rate, n_fft, n_mels).to(self.device)
        mel_spec = torch.matmul(mel_basis, magnitudes)

        # Log mel
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        # Shape: (n_mels, time) -> (batch, n_mels, time)
        return mel_spec.unsqueeze(0)

    def _mel_filterbank(self, sr, n_fft, n_mels):
        """Create mel filterbank matrix."""
        import torch

        f_min, f_max = 0.0, sr / 2.0

        # Mel scale conversion
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

    def _transcribe(self, audio):
        """Transcribe audio to text using TensorRT."""
        import torch

        try:
            # Compute features
            features = self._compute_features(audio)  # (1, 128, time)

            # Run encoder
            encoder_inputs = {
                "audio_signal": features.to(torch.float16 if features.dtype != torch.float16 else features.dtype),
                "length": torch.tensor([features.shape[2]], dtype=torch.int64, device=self.device)
            }
            encoder_out = self.encoder.infer(encoder_inputs)

            # Get encoder outputs - find the right key
            enc_key = [k for k in encoder_out.keys() if 'output' in k.lower() or 'encoded' in k.lower()]
            if enc_key:
                encoded = encoder_out[enc_key[0]]
            else:
                encoded = list(encoder_out.values())[0]

        except Exception as e:
            pb_utils.Logger.log_error(f"Encoder error: {e}")
            return ""

        # Greedy decode
        tokens = self._greedy_decode(encoded)
        return self._tokens_to_text(tokens)

    def _greedy_decode(self, encoder_out):
        """Greedy transducer decoding with decoder_joint."""
        import torch

        if encoder_out.dim() == 2:
            encoder_out = encoder_out.unsqueeze(0)

        batch_size, num_frames, enc_dim = encoder_out.shape
        tokens = []

        # Initialize decoder states (2 LSTM layers, hidden_size=640)
        hidden_size = 640
        state_1 = torch.zeros(2, 1, hidden_size, dtype=encoder_out.dtype, device=self.device)
        state_2 = torch.zeros(2, 1, hidden_size, dtype=encoder_out.dtype, device=self.device)

        # Current target (start with blank)
        target = torch.tensor([[self.blank_id]], dtype=torch.int64, device=self.device)
        target_len = torch.tensor([1], dtype=torch.int64, device=self.device)

        for t in range(num_frames):
            # Get single frame (1, 1, enc_dim) -> need (1, enc_dim, 1) based on model
            enc_frame = encoder_out[:, t:t+1, :].transpose(1, 2)  # (1, enc_dim, 1)

            for _ in range(10):  # Max symbols per frame
                try:
                    decoder_inputs = {
                        "encoder_outputs": enc_frame.contiguous(),
                        "targets": target,
                        "target_length": target_len,
                        "input_states_1": state_1.contiguous(),
                        "input_states_2": state_2.contiguous(),
                    }

                    outputs = self.decoder_joint.infer(decoder_inputs)

                    # Get logits and states
                    logits = outputs.get("outputs", list(outputs.values())[0])
                    state_1 = outputs.get("output_states_1", state_1)
                    state_2 = outputs.get("output_states_2", state_2)

                    # Get token
                    token_id = int(torch.argmax(logits.flatten()).item())

                except Exception as e:
                    pb_utils.Logger.log_error(f"Decoder error: {e}")
                    break

                if token_id == self.blank_id:
                    break
                else:
                    tokens.append(token_id)
                    target = torch.tensor([[token_id]], dtype=torch.int64, device=self.device)

        return tokens

    def _tokens_to_text(self, tokens):
        """Convert tokens to text."""
        pieces = []
        for tid in tokens:
            if tid in self.vocab:
                token = self.vocab[tid]
                if token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>", "<blk>"]:
                    pieces.append(token)

        text = "".join(pieces).replace("▁", " ").strip()
        return text

    def finalize(self):
        pass
