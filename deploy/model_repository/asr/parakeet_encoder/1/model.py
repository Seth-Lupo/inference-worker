"""
Parakeet Encoder - ONNX Runtime Python Backend

Loads the encoder ONNX model and runs inference on GPU using ONNX Runtime.
Called by parakeet_tdt BLS orchestrator.

Input: Mel spectrogram features [1, 128, time]
Output: Encoded features [1, time/4, 512]
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Parakeet Encoder using ONNX Runtime GPU."""

    def initialize(self, args):
        """Load ONNX model with ONNX Runtime."""
        import onnxruntime as ort

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        # Get ONNX model path from parameters or use default
        params = self.model_config.get("parameters", {})
        onnx_dir = params.get("ONNX_DIR", {}).get("string_value", "/models/asr/parakeet_onnx")
        onnx_path = os.path.join(onnx_dir, "encoder-model.onnx")

        pb_utils.Logger.log_info(f"Parakeet Encoder: Loading ONNX from {onnx_path}")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Create ONNX Runtime session with CUDA EP
        # Minimal memory footprint to coexist with PyTorch models loading in parallel
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 256 * 1024 * 1024,  # 256MB
                'cudnn_conv_use_max_workspace': False,
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.enable_mem_reuse = True

        # Retry with delay to handle GPU memory contention during parallel model loading
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
                break
            except Exception as e:
                if attempt < max_retries - 1 and "memory" in str(e).lower():
                    pb_utils.Logger.log_info(f"Parakeet Encoder: GPU memory contention, retry {attempt + 1}/{max_retries}")
                    time.sleep(2)
                else:
                    raise

        # Log model info
        pb_utils.Logger.log_info(f"Parakeet Encoder: Providers = {self.session.get_providers()}")
        for inp in self.session.get_inputs():
            pb_utils.Logger.log_info(f"  Input: {inp.name} {inp.shape} {inp.type}")
        for out in self.session.get_outputs():
            pb_utils.Logger.log_info(f"  Output: {out.name} {out.shape} {out.type}")

        pb_utils.Logger.log_info("Parakeet Encoder: Initialized")

    def execute(self, requests):
        """Run ONNX inference."""
        responses = []

        for request in requests:
            # Get inputs
            audio_signal = pb_utils.get_input_tensor_by_name(request, "audio_signal")
            length = pb_utils.get_input_tensor_by_name(request, "length")

            audio_np = audio_signal.as_numpy().astype(np.float16)
            length_np = length.as_numpy().astype(np.int64)

            try:
                # Run ONNX inference
                outputs = self.session.run(
                    None,
                    {
                        "audio_signal": audio_np,
                        "length": length_np
                    }
                )

                # outputs[0] = encoded features, outputs[1] = output length
                encoder_out = outputs[0].astype(np.float16)
                encoder_len = outputs[1].astype(np.int64)

                out_tensor = pb_utils.Tensor("outputs", encoder_out)
                len_tensor = pb_utils.Tensor("outputs_length", encoder_len)

                responses.append(pb_utils.InferenceResponse([out_tensor, len_tensor]))

            except Exception as e:
                pb_utils.Logger.log_error(f"Encoder error: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self):
        """Cleanup."""
        pb_utils.Logger.log_info("Parakeet Encoder: Finalizing")
        del self.session
