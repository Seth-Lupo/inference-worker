from .t3 import T3VllmModel, SPEECH_TOKEN_OFFSET

# Register T3 model and tokenizers with vLLM
try:
    from vllm import ModelRegistry
    from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

    # Register the Chatterbox T3 model architecture
    ModelRegistry.register_model("ChatterboxT3", T3VllmModel)

    # Register custom tokenizers (relative to this package)
    TokenizerRegistry.register("EnTokenizer", "chatterbox_tts.models.t3.entokenizer", "EnTokenizer")
    TokenizerRegistry.register("MtlTokenizer", "chatterbox_tts.models.t3.mtltokenizer", "MTLTokenizer")
except ImportError as e:
    # vLLM not available - T3 won't work but other parts might
    import warnings
    warnings.warn(f"vLLM not available, T3 model registration skipped: {e}")
