"""
Parakeet Decoder Joint - ONNX Runtime Python Backend

Loads the decoder_joint ONNX model and runs inference on GPU using ONNX Runtime.
Called by parakeet_tdt BLS orchestrator for autoregressive decoding.

Transducer decoder with joint network - called once per frame during greedy decoding.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Parakeet Decoder using ONNX Runtime GPU."""

    def initialize(self, args):
        """Load ONNX model with ONNX Runtime."""
        import onnxruntime as ort

        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        # Get ONNX model path from parameters or use default
        params = self.model_config.get("parameters", {})
        onnx_dir = params.get("ONNX_DIR", {}).get("string_value", "/models/asr/parakeet_onnx")
        onnx_path = os.path.join(onnx_dir, "decoder_joint-model.onnx")

        pb_utils.Logger.log_info(f"Parakeet Decoder: Loading ONNX from {onnx_path}")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Create ONNX Runtime session with CUDA EP
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 1GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # Log model info
        pb_utils.Logger.log_info(f"Parakeet Decoder: Providers = {self.session.get_providers()}")
        for inp in self.session.get_inputs():
            pb_utils.Logger.log_info(f"  Input: {inp.name} {inp.shape} {inp.type}")
        for out in self.session.get_outputs():
            pb_utils.Logger.log_info(f"  Output: {out.name} {out.shape} {out.type}")

        pb_utils.Logger.log_info("Parakeet Decoder: Initialized")

    def execute(self, requests):
        """Run ONNX inference."""
        responses = []

        for request in requests:
            try:
                # Get inputs
                encoder_outputs = pb_utils.get_input_tensor_by_name(request, "encoder_outputs")
                targets = pb_utils.get_input_tensor_by_name(request, "targets")
                target_length = pb_utils.get_input_tensor_by_name(request, "target_length")
                input_states_1 = pb_utils.get_input_tensor_by_name(request, "input_states_1")
                input_states_2 = pb_utils.get_input_tensor_by_name(request, "input_states_2")

                enc_np = encoder_outputs.as_numpy().astype(np.float16)
                tgt_np = targets.as_numpy().astype(np.int64)
                tgt_len_np = target_length.as_numpy().astype(np.int64)
                states_1_np = input_states_1.as_numpy().astype(np.float16)
                states_2_np = input_states_2.as_numpy().astype(np.float16)

                # Run ONNX inference
                outputs = self.session.run(
                    None,
                    {
                        "encoder_outputs": enc_np,
                        "targets": tgt_np,
                        "target_length": tgt_len_np,
                        "input_states_1": states_1_np,
                        "input_states_2": states_2_np
                    }
                )

                # outputs[0] = logits, outputs[1] = new_states_1, outputs[2] = new_states_2
                logits = outputs[0].astype(np.float16)
                new_states_1 = outputs[1].astype(np.float16)
                new_states_2 = outputs[2].astype(np.float16)

                out_logits = pb_utils.Tensor("outputs", logits)
                out_states_1 = pb_utils.Tensor("output_states_1", new_states_1)
                out_states_2 = pb_utils.Tensor("output_states_2", new_states_2)

                responses.append(pb_utils.InferenceResponse([out_logits, out_states_1, out_states_2]))

            except Exception as e:
                pb_utils.Logger.log_error(f"Decoder error: {e}")
                import traceback
                traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self):
        """Cleanup."""
        pb_utils.Logger.log_info("Parakeet Decoder: Finalizing")
        del self.session
