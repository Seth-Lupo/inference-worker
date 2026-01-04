"""
Parakeet TDT 0.6B V2 - BLS Orchestrator

Business Logic Scripting model that orchestrates:
- parakeet_encoder (native TensorRT backend)
- parakeet_decoder (native TensorRT backend)

This Python model only handles:
1. Audio preprocessing (mel spectrogram)
2. Calling encoder ONCE via Triton internal API
3. Autoregressive decoding loop calling decoder
4. Token to text conversion

The heavy compute (encoder, decoder) runs on native TensorRT backends
for maximum performance.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Parakeet TDT ASR - BLS orchestrating native TRT backends."""

    def initialize(self, args):
        """Load vocabulary and prepare for inference."""
        import torch

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16  # Match TRT engine precision

        pb_utils.Logger.log_info(f"Parakeet BLS: Initializing")
        pb_utils.Logger.log_info(f"  Model dir: {model_dir}")
        pb_utils.Logger.log_info(f"  Device: {self.device}")

        # Load vocabulary
        vocab_path = os.path.join(model_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(model_dir, "tokens.txt")
        self.vocab = self._load_vocab(vocab_path)
        self.blank_id = 0
        self.vocab_size = len(self.vocab)

        pb_utils.Logger.log_info(f"  Vocab size: {self.vocab_size}")

        # Mel spectrogram parameters
        self.sample_rate = 16000
        self.n_fft = 512
        self.hop_length = 160
        self.n_mels = 128

        # Precompute mel filterbank
        self.mel_basis = self._mel_filterbank(
            self.sample_rate, self.n_fft, self.n_mels
        ).to(self.device).to(self.dtype)

        # LSTM hidden size for decoder states
        self.hidden_size = 640

        pb_utils.Logger.log_info("Parakeet BLS: Initialized - ready to call native TRT backends")

    def _load_vocab(self, path):
        """Load token vocabulary."""
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    vocab[i] = token
        return vocab

    def _mel_filterbank(self, sr, n_fft, n_mels):
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

    def _compute_features(self, audio):
        """Compute mel spectrogram features."""
        import torch
        import torch.nn.functional as F

        # Convert to tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        audio = audio.to(self.device).to(torch.float32)

        # Pad audio
        pad_amount = self.n_fft // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')

        # STFT
        window = torch.hann_window(self.n_fft, device=self.device)
        stft = torch.stft(audio, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft.abs() ** 2

        # Mel spectrogram
        mel_spec = torch.matmul(self.mel_basis.float(), magnitudes)
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        # Shape: (n_mels, time) -> (1, n_mels, time)
        return mel_spec.unsqueeze(0).to(self.dtype)

    def execute(self, requests):
        """Run inference using BLS to call native TRT backends."""
        import torch

        responses = []

        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio")
            audio = audio.as_numpy().flatten().astype(np.float32)

            # Normalize
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            try:
                text = self._transcribe_bls(audio)
            except Exception as e:
                pb_utils.Logger.log_error(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()
                text = ""

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def _transcribe_bls(self, audio):
        """Transcribe using BLS calls to native TRT backends."""
        import torch

        # Step 1: Compute mel features (lightweight, stays in Python)
        features = self._compute_features(audio)  # (1, 128, time)
        seq_len = features.shape[2]

        # Step 2: Call encoder via BLS (native TRT backend)
        encoder_out, encoder_len = self._call_encoder(features, seq_len)

        # Step 3: Autoregressive decoding loop (calls native TRT decoder)
        tokens = self._greedy_decode_bls(encoder_out)

        # Step 4: Convert tokens to text
        return self._tokens_to_text(tokens)

    def _call_encoder(self, features, seq_len):
        """Call parakeet_encoder model via BLS."""
        import torch

        # Prepare inputs for encoder
        audio_signal = pb_utils.Tensor(
            "audio_signal",
            features.cpu().numpy()
        )
        length = pb_utils.Tensor(
            "length",
            np.array([seq_len], dtype=np.int64)
        )

        # Create inference request for encoder
        infer_request = pb_utils.InferenceRequest(
            model_name="parakeet_encoder",
            requested_output_names=["outputs", "outputs_length"],
            inputs=[audio_signal, length]
        )

        # Execute synchronously
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise RuntimeError(f"Encoder error: {infer_response.error().message()}")

        # Get outputs
        outputs = pb_utils.get_output_tensor_by_name(infer_response, "outputs")
        outputs_length = pb_utils.get_output_tensor_by_name(infer_response, "outputs_length")

        encoder_out = torch.from_numpy(outputs.as_numpy()).to(self.device)
        encoder_len = outputs_length.as_numpy()[0]

        return encoder_out, encoder_len

    def _greedy_decode_bls(self, encoder_out):
        """Greedy transducer decoding calling native TRT decoder."""
        import torch

        # encoder_out: (1, time, 512)
        if encoder_out.dim() == 2:
            encoder_out = encoder_out.unsqueeze(0)

        batch_size, num_frames, enc_dim = encoder_out.shape
        tokens = []

        # Initialize decoder states
        state_1 = torch.zeros(2, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        state_2 = torch.zeros(2, 1, self.hidden_size, dtype=self.dtype, device=self.device)

        # Current target (start with blank)
        target = np.array([[self.blank_id]], dtype=np.int64)
        target_len = np.array([1], dtype=np.int64)

        for t in range(num_frames):
            # Get single frame: (1, time, enc_dim) -> (1, enc_dim, 1)
            enc_frame = encoder_out[:, t:t+1, :].transpose(1, 2).contiguous()

            for _ in range(10):  # Max symbols per frame
                # Call decoder via BLS
                logits, state_1, state_2 = self._call_decoder(
                    enc_frame, target, target_len, state_1, state_2
                )

                # Get token
                token_id = int(torch.argmax(logits.flatten()).item())

                if token_id == self.blank_id:
                    break
                else:
                    tokens.append(token_id)
                    target = np.array([[token_id]], dtype=np.int64)

        return tokens

    def _call_decoder(self, enc_frame, target, target_len, state_1, state_2):
        """Call parakeet_decoder model via BLS."""
        import torch

        # Prepare inputs
        encoder_outputs = pb_utils.Tensor("encoder_outputs", enc_frame.cpu().numpy())
        targets = pb_utils.Tensor("targets", target)
        target_length = pb_utils.Tensor("target_length", target_len)
        input_states_1 = pb_utils.Tensor("input_states_1", state_1.cpu().numpy())
        input_states_2 = pb_utils.Tensor("input_states_2", state_2.cpu().numpy())

        # Create inference request
        infer_request = pb_utils.InferenceRequest(
            model_name="parakeet_decoder",
            requested_output_names=["outputs", "output_states_1", "output_states_2"],
            inputs=[encoder_outputs, targets, target_length, input_states_1, input_states_2]
        )

        # Execute
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise RuntimeError(f"Decoder error: {infer_response.error().message()}")

        # Get outputs
        outputs = pb_utils.get_output_tensor_by_name(infer_response, "outputs")
        out_states_1 = pb_utils.get_output_tensor_by_name(infer_response, "output_states_1")
        out_states_2 = pb_utils.get_output_tensor_by_name(infer_response, "output_states_2")

        logits = torch.from_numpy(outputs.as_numpy()).to(self.device)
        new_state_1 = torch.from_numpy(out_states_1.as_numpy()).to(self.device)
        new_state_2 = torch.from_numpy(out_states_2.as_numpy()).to(self.device)

        return logits, new_state_1, new_state_2

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
        """Cleanup."""
        pb_utils.Logger.log_info("Parakeet BLS: Finalizing")
