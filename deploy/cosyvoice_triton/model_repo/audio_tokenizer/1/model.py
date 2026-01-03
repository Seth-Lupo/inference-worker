# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import torch
from torch.utils.dlpack import to_dlpack

import triton_python_backend_utils as pb_utils

import os
import queue
import logging
import numpy as np
import s3tokenizer

torch.set_num_threads(1)
ORIGINAL_VOCAB_SIZE = 151663

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrtContextWrapper:
    """TensorRT execution context pool for concurrent inference."""
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, f'failed to create trt context'
            self.trt_context_pool.put([trt_context, trt_stream])

    def acquire(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release(self, context, stream):
        self.trt_context_pool.put([context, stream])


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    """Convert ONNX model to TensorRT engine."""
    import tensorrt as trt
    logger.info("Converting ONNX to TensorRT...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError(f'failed to parse {onnx_model}')
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    for i in range(network.num_inputs):
        network.get_input(i).dtype = tensor_dtype
    for i in range(network.num_outputs):
        network.get_output(i).dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logger.info("Successfully converted ONNX to TensorRT")


class TrtS3Tokenizer:
    """TensorRT wrapper for S3Tokenizer model."""

    def __init__(self, trt_path, onnx_path, device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16

        # Convert ONNX to TRT if needed
        if not os.path.exists(trt_path) or os.path.getsize(trt_path) == 0:
            trt_kwargs = self._get_trt_kwargs()
            convert_onnx_to_trt(trt_path, trt_kwargs, onnx_path, fp16)

        # Load TRT engine
        import tensorrt as trt
        with open(trt_path, 'rb') as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert engine is not None, f'failed to load TRT engine: {trt_path}'
        self.trt_wrapper = TrtContextWrapper(engine, trt_concurrent=1, device=device)
        logger.info(f"Loaded TensorRT engine: {trt_path}")

    def _get_trt_kwargs(self):
        # S3Tokenizer input: (batch, n_mels=128, time)
        min_shape = [(1, 128, 10)]
        opt_shape = [(1, 128, 500)]
        max_shape = [(1, 128, 3000)]
        input_names = ["mel"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def quantize(self, mels, mels_lens):
        """Quantize mel spectrograms to speech codes using TensorRT."""
        [trt_context, stream], trt_engine = self.trt_wrapper.acquire()

        try:
            with torch.cuda.device(self.device):
                torch.cuda.current_stream().synchronize()
                if self.fp16:
                    mels = mels.half()
                mels = mels.to(self.device).contiguous()
                batch_size, n_mels, time_len = mels.shape

                with stream:
                    trt_context.set_input_shape('mel', (batch_size, n_mels, time_len))
                    # Output shape: (batch, time)
                    codes = torch.empty((batch_size, time_len), dtype=torch.int64, device=self.device)

                    trt_context.set_tensor_address(trt_engine.get_tensor_name(0), mels.data_ptr())
                    trt_context.set_tensor_address(trt_engine.get_tensor_name(1), codes.data_ptr())

                    assert trt_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                    torch.cuda.current_stream().synchronize()
        finally:
            self.trt_wrapper.release(trt_context, stream)

        codes_lens = mels_lens.clone() if isinstance(mels_lens, torch.Tensor) else torch.tensor(mels_lens)
        return codes, codes_lens


class TritonPythonModel:
    """Triton Python model for audio tokenization.

    This model takes reference audio input and extracts semantic tokens
    using s3tokenizer with TensorRT backend.
    """

    def initialize(self, args):
        """Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        model_dir = model_params["model_dir"]
        onnx_path = os.path.join(model_dir, "speech_tokenizer_v2.onnx")
        trt_path = os.path.join(model_dir, "speech_tokenizer_v2.fp16.trt")

        # Use TensorRT for GPU inference (converts on first load)
        self.audio_tokenizer = TrtS3Tokenizer(trt_path, onnx_path, device=self.device, fp16=True)
        logger.info("Audio tokenizer initialized with TensorRT")

    def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing tokenized outputs
        """
        mels = []

        # Process each request in batch
        for request in requests:
            # Extract input tensors
            wav_array = pb_utils.get_input_tensor_by_name(
                request, "reference_wav").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(
                request, "reference_wav_len").as_numpy().item()

            wav_array = torch.from_numpy(wav_array).to(self.device)
            # Prepare inputs
            wav = wav_array[:, :wav_len].squeeze(0)
            mels.append(s3tokenizer.log_mel_spectrogram(wav))

        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.audio_tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))
        codes = codes.clone() + ORIGINAL_VOCAB_SIZE

        responses = []
        for i in range(len(requests)):
            prompt_speech_tokens = codes[i, :codes_lens[i].item()]
            prompt_speech_tokens_tensor = pb_utils.Tensor.from_dlpack(
                "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[prompt_speech_tokens_tensor])
            responses.append(inference_response)

        return responses
