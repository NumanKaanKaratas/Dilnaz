from transformers.configuration_utils import PretrainedConfig


class NazConfig(PretrainedConfig):
    model_type = "naz"

    def __init__(
        self,
        dil_path=None,
        byte_vocab_size=256,
        vocab_size=355,
        pad_token_id=256,
        eos_token_id=257,
        max_word_bytes=32,
        latent_size=128,
        num_mlp_layers=4,
        num_samples=8,
        energy_target_samples=100,
        beta=1.0,
        pred_log_std_min=-4.0,
        pred_log_std_max=2.0,
        log_std_loss_weight=0.05,
        decode_chunk_size=512,
        semantic_feedback="mean",
        hidden_size=512,
        intermediate_size=2752,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        full_attention_interval=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_conv_kernel_size=4,
        partial_rotary_factor=0.25,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        bos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs,
    ):
        self.dil_path = dil_path
        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.max_word_bytes = max_word_bytes
        self.latent_size = latent_size
        self.num_mlp_layers = num_mlp_layers
        self.num_samples = num_samples
        self.energy_target_samples = energy_target_samples
        self.beta = beta
        self.pred_log_std_min = pred_log_std_min
        self.pred_log_std_max = pred_log_std_max
        self.log_std_loss_weight = log_std_loss_weight
        self.decode_chunk_size = decode_chunk_size
        self.semantic_feedback = semantic_feedback
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.full_attention_interval = full_attention_interval
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_conv_kernel_size = linear_conv_kernel_size
        self.partial_rotary_factor = partial_rotary_factor
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

