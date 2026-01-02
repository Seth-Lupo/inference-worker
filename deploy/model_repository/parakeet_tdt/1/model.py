"""
Parakeet TDT 0.6B V2 - Triton Python Backend

Uses TensorRT engines for GPU inference.
Falls back to PyTorch+CUDA if engines not built.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TensorRTInference:
    """TensorRT engine wrapper."""

    def __init__(self, engine_path, device_id=0):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

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

    def __call__(self, *inputs):
        import torch

        # Allocate outputs
        outputs = []
        output_tensors = {}

        for i, name in enumerate(self.input_names):
            if i < len(inputs):
                inp = inputs[i]
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp).cuda()
                self.context.set_input_shape(name, tuple(inp.shape))
                self.context.set_tensor_address(name, inp.data_ptr())

        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.bindings[name]["dtype"]
            out = torch.empty(tuple(shape), dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype, device="cuda")
            output_tensors[name] = out
            self.context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)

        # Execute
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return outputs[0] if len(outputs) == 1 else outputs


class TritonPythonModel:
    """Parakeet TDT ASR using TensorRT GPU."""

    def initialize(self, args):
        """Load TensorRT engines or fall back to PyTorch."""
        import torch

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))
        engine_dir = os.path.join(model_dir, "engines")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pb_utils.Logger.log_info(f"Loading Parakeet from {model_dir}")
        pb_utils.Logger.log_info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            pb_utils.Logger.log_info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Check for TensorRT engines
        encoder_engine = os.path.join(engine_dir, "encoder.plan")
        decoder_engine = os.path.join(engine_dir, "decoder.plan")
        joiner_engine = os.path.join(engine_dir, "joiner.plan")

        if all(os.path.exists(p) for p in [encoder_engine, decoder_engine, joiner_engine]):
            pb_utils.Logger.log_info("Loading TensorRT engines...")
            self.use_trt = True
            self.encoder = TensorRTInference(encoder_engine)
            self.decoder = TensorRTInference(decoder_engine)
            self.joiner = TensorRTInference(joiner_engine)
            pb_utils.Logger.log_info("TensorRT engines loaded")
        else:
            pb_utils.Logger.log_info("TensorRT engines not found, using PyTorch+onnx2torch")
            self.use_trt = False
            self._load_pytorch(model_dir)

        # Load vocabulary
        self.vocab = self._load_vocab(os.path.join(model_dir, "tokens.txt"))
        self.blank_id = 0

        backend = "TensorRT" if self.use_trt else "PyTorch"
        pb_utils.Logger.log_info(f"Parakeet initialized: vocab={len(self.vocab)}, backend={backend}, device={self.device}")

    def _load_pytorch(self, model_dir):
        """Fallback to PyTorch via onnx2torch."""
        import torch
        import onnx
        from onnx2torch import convert

        pb_utils.Logger.log_info("Loading encoder (PyTorch)...")
        encoder_onnx = onnx.load(os.path.join(model_dir, "encoder.onnx"))
        self.encoder = convert(encoder_onnx).to(self.device).eval()

        pb_utils.Logger.log_info("Loading decoder (PyTorch)...")
        decoder_onnx = onnx.load(os.path.join(model_dir, "decoder.onnx"))
        self.decoder = convert(decoder_onnx).to(self.device).eval()

        pb_utils.Logger.log_info("Loading joiner (PyTorch)...")
        joiner_onnx = onnx.load(os.path.join(model_dir, "joiner.onnx"))
        self.joiner = convert(joiner_onnx).to(self.device).eval()

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

            # Normalize
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            with torch.no_grad():
                text = self._transcribe(audio)

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def _transcribe(self, audio):
        """Transcribe audio to text."""
        import torch

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        # Encoder
        try:
            if self.use_trt:
                encoder_out = self.encoder(audio_tensor)
            else:
                encoder_out = self.encoder(audio_tensor)
                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]
        except Exception as e:
            pb_utils.Logger.log_error(f"Encoder error: {e}")
            return ""

        # Greedy decode
        tokens = self._greedy_decode(encoder_out)
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

            for _ in range(10):
                try:
                    if self.use_trt:
                        decoder_out = self.decoder(decoder_input)
                        joiner_out = self.joiner(enc_frame, decoder_out)
                    else:
                        decoder_out = self.decoder(decoder_input)
                        if isinstance(decoder_out, tuple):
                            decoder_out = decoder_out[0]
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
        """Convert tokens to text."""
        pieces = []
        for tid in tokens:
            if tid in self.vocab:
                token = self.vocab[tid]
                if token not in ["<blank>", "<unk>", "<s>", "</s>", "<pad>"]:
                    pieces.append(token)

        text = "".join(pieces).replace("▁", " ").strip()
        return text

    def finalize(self):
        pass
