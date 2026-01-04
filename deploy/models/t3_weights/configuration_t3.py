"""T3 Configuration for HuggingFace/vLLM compatibility."""
from transformers import PretrainedConfig


class T3Config(PretrainedConfig):
    """Configuration class for T3 (Chatterbox speech token generator).

    T3 is a Llama-based model fine-tuned for text-to-speech token generation.
    It generates discrete speech tokens that are decoded by S3Gen vocoder.
    """
    model_type = "t3"

    def __init__(
        self,
        vocab_size: int = 8194,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: dict = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        # T3-specific
        speech_vocab_size: int = 6561,
        text_vocab_size: int = 704,
        start_speech_token: int = 6561,
        stop_speech_token: int = 6562,
        conditioning_size: int = 34,
        conditioning_path: str = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # T3-specific
        self.speech_vocab_size = speech_vocab_size
        self.text_vocab_size = text_vocab_size
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.conditioning_size = conditioning_size
        self.conditioning_path = conditioning_path

        # Remove token IDs from kwargs if present to avoid duplicate argument error
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=start_speech_token,
            eos_token_id=stop_speech_token,
            **kwargs,
        )
