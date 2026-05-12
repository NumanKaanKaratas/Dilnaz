from transformers.configuration_utils import PretrainedConfig


class NazConfig(PretrainedConfig):
    model_type = "naz"

    def __init__(
        self,
        dil_path=None,
        byte_vocab_size=256,
        vocab_size=778,
        pad_token_id=256,
        eos_token_id=257,
        latent_size=512,
        reconstruction_loss_weight=1.0,
        num_semantic_candidates=4,
        mtp_horizons=3,
        mtp_loss_weights=(1.0, 0.3, 0.15),
        mixture_sigma=1.1,
        mixture_sigma_min=0.35,
        mixture_sigma_max=2.0,
        usage_balance_weight=0.05,
        router_responsibility_weight=1.0,
        moe_num_experts=8,
        moe_top_k=2,
        moe_layers=4,
        moe_balance_weight=0.01,
        moe_expert_intermediate_size=None,
        naz_input_jitter_prob=0.10,
        naz_input_jitter_min_cos=0.985,
        naz_input_jitter_max_cos=0.995,
        repetition_cos_threshold=0.985,
        min_new_tokens=1,
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
        if ("max_" + "word_bytes") in kwargs:
            raise ValueError("fixed-width surface config is not supported by Naz")
        self.dil_path = dil_path
        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.num_semantic_candidates = num_semantic_candidates
        self.mtp_horizons = mtp_horizons
        self.mtp_loss_weights = tuple(float(weight) for weight in mtp_loss_weights)
        if mixture_sigma_min <= 0.0 or mixture_sigma_max <= mixture_sigma_min:
            raise ValueError("mixture sigma bounds must satisfy 0 < min < max")
        if mixture_sigma <= mixture_sigma_min or mixture_sigma >= mixture_sigma_max:
            raise ValueError("mixture_sigma must be inside (mixture_sigma_min, mixture_sigma_max)")
        self.mixture_sigma = mixture_sigma
        self.mixture_sigma_min = mixture_sigma_min
        self.mixture_sigma_max = mixture_sigma_max
        self.usage_balance_weight = usage_balance_weight
        self.router_responsibility_weight = router_responsibility_weight
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_layers = moe_layers
        self.moe_balance_weight = moe_balance_weight
        self.moe_expert_intermediate_size = (
            intermediate_size if moe_expert_intermediate_size is None else moe_expert_intermediate_size
        )
        if naz_input_jitter_prob < 0.0 or naz_input_jitter_prob > 1.0:
            raise ValueError("naz_input_jitter_prob must be inside [0, 1]")
        if naz_input_jitter_min_cos <= 0.0 or naz_input_jitter_max_cos > 1.0 or naz_input_jitter_min_cos > naz_input_jitter_max_cos:
            raise ValueError("naz input jitter cosine range must satisfy 0 < min <= max <= 1")
        self.naz_input_jitter_prob = naz_input_jitter_prob
        self.naz_input_jitter_min_cos = naz_input_jitter_min_cos
        self.naz_input_jitter_max_cos = naz_input_jitter_max_cos
        self.repetition_cos_threshold = repetition_cos_threshold
        self.min_new_tokens = min_new_tokens
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
