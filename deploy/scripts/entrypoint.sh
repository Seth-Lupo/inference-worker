#!/bin/bash
# Entrypoint script that patches vLLM model registry before starting Triton

# Register T3 model with vLLM
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/models/t3_weights')

# Import T3 model components
from configuration_t3 import T3Config
from modeling_t3 import T3ForCausalLM

# Register with vLLM model registry
try:
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("T3ForCausalLM", T3ForCausalLM)
    print("[entrypoint] Registered T3ForCausalLM with vLLM ModelRegistry")
except Exception as e1:
    try:
        from vllm.model_executor.models import _MODELS
        _MODELS["T3ForCausalLM"] = T3ForCausalLM
        print("[entrypoint] Registered T3ForCausalLM with vLLM _MODELS dict")
    except Exception as e2:
        print(f"[entrypoint] WARNING: Could not register T3: {e1}, {e2}")

# Also register the config
try:
    from transformers import AutoConfig
    AutoConfig.register("t3", T3Config)
    print("[entrypoint] Registered T3Config with transformers AutoConfig")
except Exception as e:
    print(f"[entrypoint] WARNING: Could not register T3Config: {e}")
PYTHON_SCRIPT

# Execute Triton with all arguments passed to this script
exec tritonserver "$@"
