"""
Parakeet TDT 0.6B V2 - Triton Python Backend
Wraps ONNX model using sherpa-onnx for transducer-based ASR.

This runs on GPU via CUDA provider for fast inference.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python backend for Parakeet TDT ASR."""

    def initialize(self, args):
        """Load the sherpa-onnx recognizer."""
        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        # Try to import sherpa_onnx
        try:
            import sherpa_onnx
            self._use_sherpa = True
        except ImportError:
            pb_utils.Logger.log_warn(
                "sherpa_onnx not found, falling back to onnxruntime"
            )
            self._use_sherpa = False

        if self._use_sherpa:
            self._init_sherpa(model_dir)
        else:
            self._init_onnxruntime(model_dir)

        pb_utils.Logger.log_info("Parakeet TDT initialized")

    def _init_sherpa(self, model_dir):
        """Initialize using sherpa-onnx (handles transducer decoding)."""
        import sherpa_onnx

        # Check for transducer format (encoder/decoder/joiner)
        encoder_path = os.path.join(model_dir, "encoder.onnx")
        decoder_path = os.path.join(model_dir, "decoder.onnx")
        joiner_path = os.path.join(model_dir, "joiner.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if all(os.path.exists(p) for p in [encoder_path, decoder_path, joiner_path, tokens_path]):
            pb_utils.Logger.log_info("Using transducer model format")
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                tokens=tokens_path,
                num_threads=4,
                provider="cuda",
            )
        else:
            # Try single model format
            model_path = os.path.join(model_dir, "model.onnx")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, "model_fp16.onnx")

            tokens_path = os.path.join(model_dir, "tokens.txt")
            if not os.path.exists(tokens_path):
                # Try to find vocab.json and convert
                vocab_path = os.path.join(model_dir, "vocab.json")
                if os.path.exists(vocab_path):
                    self._convert_vocab_to_tokens(vocab_path, tokens_path)

            pb_utils.Logger.log_info(f"Using single model format: {model_path}")
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=model_path,
                tokens=tokens_path,
                num_threads=4,
                provider="cuda",
            )

    def _init_onnxruntime(self, model_dir):
        """Fallback to raw onnxruntime (limited functionality)."""
        import onnxruntime as ort

        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "encoder.onnx")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Load vocabulary for decoding
        vocab_path = os.path.join(model_dir, "vocab.json")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        self.vocab = {}
        if os.path.exists(vocab_path):
            with open(vocab_path) as f:
                self.vocab = json.load(f)
        elif os.path.exists(tokens_path):
            with open(tokens_path) as f:
                for idx, line in enumerate(f):
                    token = line.strip().split()[0] if line.strip() else ""
                    self.vocab[idx] = token

        self.recognizer = None

    def _convert_vocab_to_tokens(self, vocab_path, tokens_path):
        """Convert vocab.json to tokens.txt format."""
        with open(vocab_path) as f:
            vocab = json.load(f)

        with open(tokens_path, "w") as f:
            for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{token} {idx}\n")

    def execute(self, requests):
        """Run inference on batch of audio inputs."""
        responses = []

        for request in requests:
            # Get input tensors
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio")
            if audio_tensor is None:
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_signal")

            audio = audio_tensor.as_numpy().flatten().astype(np.float32)

            # Normalize if needed (should be float32 in range [-1, 1])
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / 32768.0

            if self._use_sherpa and self.recognizer:
                # Use sherpa-onnx for full transcription
                stream = self.recognizer.create_stream()
                stream.accept_waveform(16000, audio)
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
            else:
                # Fallback: raw ONNX inference (returns tokens, basic decoding)
                text = self._infer_onnxruntime(audio)

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "transcription",
                np.array([text], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def _infer_onnxruntime(self, audio):
        """Run inference with raw onnxruntime and decode tokens."""
        # Prepare inputs
        audio = audio.reshape(1, -1)
        length = np.array([[audio.shape[1]]], dtype=np.int64)

        # Run inference
        inputs = {
            "audio_signal": audio,
            "length": length,
        }

        try:
            outputs = self.session.run(None, inputs)
            tokens = outputs[0].flatten()

            # Decode tokens to text
            text_parts = []
            for token_id in tokens:
                if token_id in self.vocab:
                    token = self.vocab[token_id]
                    if token and token not in ["<blank>", "<pad>", "<unk>", "<s>", "</s>"]:
                        text_parts.append(token)

            text = "".join(text_parts).replace("▁", " ").strip()
            return text

        except Exception as e:
            pb_utils.Logger.log_error(f"Inference error: {e}")
            return ""

    def finalize(self):
        """Cleanup."""
        pass
