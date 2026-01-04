"""T3 Model Configuration for HuggingFace/vLLM compatibility."""
from transformers import PretrainedConfig


class T3Config(PretrainedConfig):
    """
    Configuration for T3 (Text-to-Token Transformer) model.

    T3 is a GPT2-style autoregressive model that generates speech tokens
    from text tokens. Used in Chatterbox TTS.
    """
    model_type = "t3"

    def __init__(
        self,
        vocab_size: int = 8194,
        text_vocab_size: int = 704,
        speech_vocab_size: int = 8194,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 24,
        n_positions: int = 4096,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 4096,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        tie_word_embeddings: bool = False,
        speaker_embed_dim: int = 256,
        cond_channels: int = 768,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.speaker_embed_dim = speaker_embed_dim
        self.cond_channels = cond_channels

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
