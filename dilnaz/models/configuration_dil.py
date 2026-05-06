from transformers.configuration_utils import PretrainedConfig


class DilConfig(PretrainedConfig):
    model_type = "dil"

    def __init__(
        self,
        byte_vocab_size=256,
        vocab_size=778,
        pad_token_id=256,
        eos_token_id=257,
        hidden_size=512,
        intermediate_size=1280,
        num_encoder_layers=2,
        num_decoder_layers=2,
        latent_size=128,
        max_word_bytes=32,
        context_radius=2,
        dil_dropout=0.15,
        kl_clamp=0.5,
        kl_weight=1e-3,
        ce_weight=1.0,
        distillation_weight=16.0,
        layer_geometry_weight=4.0,
        mean_geometry_weight=8.0,
        variance_weight=0.05,
        semantic_normalizer_momentum=0.01,
        semantic_normalizer_eps=1e-4,
        semantic_normalizer_z_clip=6.0,
        normalized_log_std_min=-8.0,
        normalized_log_std_max=4.0,
        decoder_start_token_id=None,
        tokenizer_vocab_file="hybrid_surface_vocab.json",
        nllb_model_name="facebook/nllb-200-distilled-600M",
        nllb_src_lang="tur_Latn",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mlp_bias=False,
        checkpoint_format_version=11,
        **kwargs,
    ):
        if "context_left_radius" in kwargs:
            raise ValueError("context_left_radius is not supported; use context_radius")
        kwargs.pop("context_size", None)
        kwargs.pop("target_index", None)
        if context_radius < 0:
            raise ValueError("context_radius must be >= 0")
        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.latent_size = latent_size
        self.max_word_bytes = max_word_bytes
        self.context_radius = context_radius
        self.context_size = context_radius * 2 + 1
        self.target_index = context_radius
        self.dil_dropout = dil_dropout
        self.kl_clamp = kl_clamp
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.distillation_weight = distillation_weight
        self.layer_geometry_weight = layer_geometry_weight
        self.mean_geometry_weight = mean_geometry_weight
        self.variance_weight = variance_weight
        self.semantic_normalizer_momentum = semantic_normalizer_momentum
        self.semantic_normalizer_eps = semantic_normalizer_eps
        self.semantic_normalizer_z_clip = semantic_normalizer_z_clip
        self.normalized_log_std_min = normalized_log_std_min
        self.normalized_log_std_max = normalized_log_std_max
        resolved_decoder_start_token_id = eos_token_id if decoder_start_token_id is None else decoder_start_token_id
        self.tokenizer_vocab_file = tokenizer_vocab_file
        self.nllb_model_name = nllb_model_name
        self.nllb_src_lang = nllb_src_lang
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.mlp_bias = mlp_bias
        self.checkpoint_format_version = checkpoint_format_version

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=resolved_decoder_start_token_id,
            **kwargs,
        )
        self.decoder_start_token_id = resolved_decoder_start_token_id
