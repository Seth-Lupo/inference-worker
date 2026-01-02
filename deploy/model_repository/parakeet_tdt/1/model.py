"""
Parakeet TDT 0.6B V2 - Triton Python Backend

Uses PyTorch with CUDA for GPU inference.
Loads ONNX models via onnx2torch for native PyTorch execution.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Parakeet TDT ASR using PyTorch GPU."""

    def initialize(self, args):
        """Load ONNX models as PyTorch on CUDA."""
        import torch
        import onnx
        from onnx2torch import convert

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        # Check CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pb_utils.Logger.log_info(f"Loading Parakeet models from {model_dir}")
        pb_utils.Logger.log_info(f"PyTorch device: {self.device}, CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            pb_utils.Logger.log_info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Load ONNX models and convert to PyTorch
        pb_utils.Logger.log_info("Loading encoder...")
        encoder_onnx = onnx.load(os.path.join(model_dir, "encoder.onnx"))
        self.encoder = convert(encoder_onnx).to(self.device).eval()

        pb_utils.Logger.log_info("Loading decoder...")
        decoder_onnx = onnx.load(os.path.join(model_dir, "decoder.onnx"))
        self.decoder = convert(decoder_onnx).to(self.device).eval()

        pb_utils.Logger.log_info("Loading joiner...")
        joiner_onnx = onnx.load(os.path.join(model_dir, "joiner.onnx"))
        self.joiner = convert(joiner_onnx).to(self.device).eval()

        # Load vocabulary
        self.vocab = self._load_vocab(os.path.join(model_dir, "tokens.txt"))
        self.blank_id = 0

        pb_utils.Logger.log_info(
            f"Parakeet initialized: vocab_size={len(self.vocab)}, device={self.device}"
        )

    def _load_vocab(self, path):
        """Load token vocabulary."""
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token, idx = parts[0], int(parts[1])
                    vocab[idx] = token
                elif len(parts) == 1:
                    vocab[len(vocab)] = parts[0]
        return vocab

    def execute(self, requests):
        """Run inference on audio."""
        import torch

        responses = []

        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio")
            audio = audio.as_numpy().flatten().astype(np.float32)

            # Normalize to [-1, 1]
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            # Transcribe
            with torch.no_grad():
                text = self._transcribe(audio)

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def _transcribe(self, audio):
        """Greedy transducer decoding with PyTorch."""
        import torch

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([[audio_tensor.shape[1]]], dtype=torch.long, device=self.device)

        # Encoder
        try:
            # Try common input names
            try:
                encoder_out = self.encoder(audio_tensor, audio_len)
            except TypeError:
                encoder_out = self.encoder(audio_tensor)

            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
        except Exception as e:
            pb_utils.Logger.log_error(f"Encoder error: {e}")
            return ""

        # Greedy decode
        tokens = self._greedy_decode(encoder_out)

        # Convert to text
        return self._tokens_to_text(tokens)

    def _greedy_decode(self, encoder_out):
        """Greedy transducer decoding."""
        import torch

        if encoder_out.dim() == 2:
            encoder_out = encoder_out.unsqueeze(0)

        batch_size, num_frames, enc_dim = encoder_out.shape
        tokens = []
        decoder_input = torch.tensor([[self.blank_id]], dtype=torch.long, device=self.device)

        for t in range(num_frames):
            enc_frame = encoder_out[:, t:t+1, :]

            for _ in range(10):  # Max tokens per frame
                try:
                    decoder_out = self.decoder(decoder_input)
                    if isinstance(decoder_out, tuple):
                        decoder_out = decoder_out[0]
                except Exception:
                    break

                try:
                    joiner_out = self.joiner(enc_frame, decoder_out)
                    if isinstance(joiner_out, tuple):
                        joiner_out = joiner_out[0]
                except Exception:
                    break

                logits = joiner_out.squeeze()
                token_id = int(torch.argmax(logits).item())

                if token_id == self.blank_id:
                    break
                else:
                    tokens.append(token_id)
                    decoder_input = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        return tokens

    def _tokens_to_text(self, tokens):
        """Convert token IDs to text."""
        pieces = []
        for tid in tokens:
            if tid in self.vocab:
                token = self.vocab[tid]
                if token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>"]:
                    pieces.append(token)

        text = "".join(pieces)
        text = text.replace("▁", " ").strip()
        return text

    def finalize(self):
        """Cleanup."""
        pass
