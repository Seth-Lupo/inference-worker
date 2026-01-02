"""
Parakeet TDT 0.6B V2 - Triton Python Backend

Uses ONNX Runtime with CUDA for GPU inference.
Implements greedy transducer decoding without sherpa-onnx.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Parakeet TDT ASR using ONNX Runtime GPU."""

    def initialize(self, args):
        """Load ONNX models with CUDA execution provider."""
        import onnxruntime as ort

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        # ONNX Runtime with CUDA
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        pb_utils.Logger.log_info(f"Loading Parakeet models from {model_dir}")
        pb_utils.Logger.log_info(f"ONNX Runtime providers: {ort.get_available_providers()}")

        # Load transducer models
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder.onnx"),
            providers=providers,
        )
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "decoder.onnx"),
            providers=providers,
        )
        self.joiner = ort.InferenceSession(
            os.path.join(model_dir, "joiner.onnx"),
            providers=providers,
        )

        # Load vocabulary
        self.vocab = self._load_vocab(os.path.join(model_dir, "tokens.txt"))
        self.blank_id = 0  # Usually 0 for transducers

        pb_utils.Logger.log_info(
            f"Parakeet initialized: vocab_size={len(self.vocab)}, "
            f"using {self.encoder.get_providers()[0]}"
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
                    # Some formats have just the token, idx is line number
                    vocab[len(vocab)] = parts[0]
        return vocab

    def execute(self, requests):
        """Run inference on audio."""
        responses = []

        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio")
            audio = audio.as_numpy().flatten().astype(np.float32)

            # Normalize to [-1, 1]
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            # Transcribe
            text = self._transcribe(audio)

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def _transcribe(self, audio):
        """
        Greedy transducer decoding.

        Transducer: P(y|x) via encoder-decoder-joiner architecture.
        """
        # Encoder: audio -> encoder_out
        # Input shape: [batch, time] or [batch, features, time]
        audio_input = audio.reshape(1, -1)
        audio_len = np.array([[audio_input.shape[1]]], dtype=np.int64)

        # Get encoder input names
        enc_inputs = {inp.name: None for inp in self.encoder.get_inputs()}

        if "audio_signal" in enc_inputs:
            enc_inputs["audio_signal"] = audio_input
        elif "input" in enc_inputs:
            enc_inputs["input"] = audio_input
        else:
            # Use first input
            enc_inputs[self.encoder.get_inputs()[0].name] = audio_input

        # Add length if required
        for inp in self.encoder.get_inputs():
            if "length" in inp.name.lower() and inp.name not in enc_inputs:
                enc_inputs[inp.name] = audio_len

        # Remove None entries
        enc_inputs = {k: v for k, v in enc_inputs.items() if v is not None}

        try:
            encoder_out = self.encoder.run(None, enc_inputs)[0]
        except Exception as e:
            pb_utils.Logger.log_error(f"Encoder error: {e}")
            return ""

        # Greedy decode
        tokens = self._greedy_decode(encoder_out)

        # Convert to text
        text = self._tokens_to_text(tokens)
        return text

    def _greedy_decode(self, encoder_out):
        """
        Greedy transducer decoding.

        At each encoder frame:
        1. Run decoder on current hypothesis
        2. Run joiner on (encoder_out, decoder_out)
        3. Take argmax of joiner output
        4. If not blank, append to hypothesis and repeat
        5. If blank, move to next encoder frame
        """
        batch_size, num_frames, enc_dim = encoder_out.shape

        # Initialize decoder state with blank
        tokens = []
        decoder_input = np.array([[self.blank_id]], dtype=np.int64)

        # Get decoder/joiner input names
        dec_input_name = self.decoder.get_inputs()[0].name
        join_enc_name = self.joiner.get_inputs()[0].name
        join_dec_name = self.joiner.get_inputs()[1].name if len(self.joiner.get_inputs()) > 1 else None

        for t in range(num_frames):
            enc_frame = encoder_out[:, t:t+1, :]  # [1, 1, enc_dim]

            # Limit iterations per frame to prevent infinite loops
            for _ in range(10):
                # Decoder
                try:
                    decoder_out = self.decoder.run(None, {dec_input_name: decoder_input})[0]
                except Exception:
                    break

                # Joiner
                try:
                    if join_dec_name:
                        joiner_out = self.joiner.run(None, {
                            join_enc_name: enc_frame,
                            join_dec_name: decoder_out,
                        })[0]
                    else:
                        # Some joiners concatenate internally
                        joiner_out = self.joiner.run(None, {join_enc_name: enc_frame})[0]
                except Exception:
                    break

                # Argmax
                logits = joiner_out.squeeze()
                token_id = int(np.argmax(logits))

                if token_id == self.blank_id:
                    # Blank: move to next frame
                    break
                else:
                    # Non-blank: emit token, update decoder input
                    tokens.append(token_id)
                    decoder_input = np.array([[token_id]], dtype=np.int64)

        return tokens

    def _tokens_to_text(self, tokens):
        """Convert token IDs to text."""
        pieces = []
        for tid in tokens:
            if tid in self.vocab:
                token = self.vocab[tid]
                # Skip special tokens
                if token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>"]:
                    pieces.append(token)

        # Join and handle SentencePiece markers
        text = "".join(pieces)
        text = text.replace("▁", " ").strip()
        return text

    def finalize(self):
        """Cleanup."""
        pass
