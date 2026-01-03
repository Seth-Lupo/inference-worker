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
            pb_utils.Logger.log_info("TensorRT engines not found, using ONNX Runtime")
            self.use_trt = False
            self._load_pytorch(model_dir)

        # Load vocabulary
        self.vocab = self._load_vocab(os.path.join(model_dir, "tokens.txt"))
        self.blank_id = 0

        backend = "TensorRT" if self.use_trt else "ONNX Runtime"
        pb_utils.Logger.log_info(f"Parakeet initialized: vocab={len(self.vocab)}, backend={backend}, device={self.device}")

    def _load_pytorch(self, model_dir):
        """Fallback to ONNX Runtime (supports INT8 quantized models)."""
        import onnxruntime as ort

        # Use CUDA execution provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        pb_utils.Logger.log_info("Loading encoder (ONNX Runtime)...")
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder.onnx"),
            sess_options=sess_options,
            providers=providers
        )

        pb_utils.Logger.log_info("Loading decoder (ONNX Runtime)...")
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "decoder.onnx"),
            sess_options=sess_options,
            providers=providers
        )

        pb_utils.Logger.log_info("Loading joiner (ONNX Runtime)...")
        self.joiner = ort.InferenceSession(
            os.path.join(model_dir, "joiner.onnx"),
            sess_options=sess_options,
            providers=providers
        )

        # Store input/output names for inference
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.encoder_output = self.encoder.get_outputs()[0].name
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.decoder_output = self.decoder.get_outputs()[0].name
        self.joiner_inputs = [inp.name for inp in self.joiner.get_inputs()]
        self.joiner_output = self.joiner.get_outputs()[0].name

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

        # Encoder
        try:
            if self.use_trt:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                encoder_out = self.encoder(audio_tensor)
            else:
                # ONNX Runtime path
                audio_input = audio.reshape(1, -1).astype(np.float32)
                encoder_out = self.encoder.run(
                    [self.encoder_output],
                    {self.encoder_input: audio_input}
                )[0]
                encoder_out = torch.from_numpy(encoder_out).to(self.device)
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
        decoder_input = np.array([[self.blank_id]], dtype=np.int64)

        for t in range(num_frames):
            enc_frame = encoder_out[:, t:t+1, :].cpu().numpy()

            for _ in range(10):
                try:
                    if self.use_trt:
                        decoder_input_t = torch.from_numpy(decoder_input).to(self.device)
                        enc_frame_t = torch.from_numpy(enc_frame).to(self.device)
                        decoder_out = self.decoder(decoder_input_t)
                        joiner_out = self.joiner(enc_frame_t, decoder_out)
                        logits = joiner_out.squeeze()
                        token_id = int(torch.argmax(logits).item())
                    else:
                        # ONNX Runtime path
                        decoder_out = self.decoder.run(
                            [self.decoder_output],
                            {self.decoder_input: decoder_input}
                        )[0]
                        joiner_out = self.joiner.run(
                            [self.joiner_output],
                            {
                                self.joiner_inputs[0]: enc_frame.astype(np.float32),
                                self.joiner_inputs[1]: decoder_out.astype(np.float32)
                            }
                        )[0]
                        logits = joiner_out.squeeze()
                        token_id = int(np.argmax(logits))
                except Exception:
                    break

                if token_id == self.blank_id:
                    break
                else:
                    tokens.append(token_id)
                    decoder_input = np.array([[token_id]], dtype=np.int64)

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
