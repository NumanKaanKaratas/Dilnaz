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
        latent_size=128,
        max_word_bytes=32,
        context_radius=2,
        byte_conv_layers=2,
        byte_conv_kernel_size=5,
        byte_conv_expansion=2,
        dil_dropout=0.15,
        distillation_weight=16.0,
        layer_geometry_weight=4.0,
        mean_geometry_weight=8.0,
        variance_weight=0.05,
        writer_loss_weight=1.0,
        writer_num_layers=2,
        writer_conv_kernel_size=5,
        writer_conv_expansion=2,
        writer_dropout=0.1,
        writer_noise_warmup_steps=1000,
        writer_noise_clean_ratio=0.10,
        writer_noise_easy_ratio=0.70,
        writer_noise_mid_ratio=0.20,
        writer_noise_hard_ratio=0.00,
        writer_noise_easy_min_cos=0.985,
        writer_noise_easy_max_cos=0.995,
        writer_noise_mid_min_cos=0.970,
        writer_noise_mid_max_cos=0.985,
        writer_noise_hard_min_cos=0.950,
        writer_noise_hard_max_cos=0.970,
        decoder_start_token_id=None,
        tokenizer_vocab_file="hybrid_surface_vocab.json",
        nllb_model_name="facebook/nllb-200-distilled-600M",
        nllb_src_lang="tur_Latn",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mlp_bias=False,
        checkpoint_format_version=21,
        **kwargs,
    ):
        if "context_left_radius" in kwargs:
            raise ValueError("context_left_radius is not supported; use context_radius")
        kwargs.pop("context_size", None)
        kwargs.pop("target_index", None)
        if context_radius < 0:
            raise ValueError("context_radius must be >= 0")
        if byte_conv_layers < 0 or writer_num_layers < 0:
            raise ValueError("conv layer counts must be >= 0")
        if byte_conv_kernel_size <= 0 or byte_conv_kernel_size % 2 == 0:
            raise ValueError("byte_conv_kernel_size must be a positive odd integer")
        if writer_conv_kernel_size <= 0 or writer_conv_kernel_size % 2 == 0:
            raise ValueError("writer_conv_kernel_size must be a positive odd integer")
        writer_noise_ratios = (
            writer_noise_clean_ratio,
            writer_noise_easy_ratio,
            writer_noise_mid_ratio,
            writer_noise_hard_ratio,
        )
        if writer_noise_warmup_steps < 0:
            raise ValueError("writer_noise_warmup_steps must be >= 0")
        if any(ratio < 0.0 for ratio in writer_noise_ratios):
            raise ValueError("writer noise ratios must be >= 0")
        if sum(writer_noise_ratios) <= 0.0:
            raise ValueError("at least one writer noise ratio must be positive")
        cosine_ranges = (
            (writer_noise_easy_min_cos, writer_noise_easy_max_cos),
            (writer_noise_mid_min_cos, writer_noise_mid_max_cos),
            (writer_noise_hard_min_cos, writer_noise_hard_max_cos),
        )
        if any(min_cos <= 0.0 or max_cos > 1.0 or min_cos > max_cos for min_cos, max_cos in cosine_ranges):
            raise ValueError("writer noise cosine ranges must satisfy 0 < min <= max <= 1")
        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.latent_size = latent_size
        self.max_word_bytes = max_word_bytes
        self.context_radius = context_radius
        self.context_size = 2 * context_radius + 1
        self.target_index = context_radius
        self.byte_conv_layers = byte_conv_layers
        self.byte_conv_kernel_size = byte_conv_kernel_size
        self.byte_conv_expansion = byte_conv_expansion
        self.dil_dropout = dil_dropout
        self.distillation_weight = distillation_weight
        self.layer_geometry_weight = layer_geometry_weight
        self.mean_geometry_weight = mean_geometry_weight
        self.variance_weight = variance_weight
        self.writer_loss_weight = writer_loss_weight
        self.writer_num_layers = writer_num_layers
        self.writer_conv_kernel_size = writer_conv_kernel_size
        self.writer_conv_expansion = writer_conv_expansion
        self.writer_dropout = writer_dropout
        self.writer_vocab_size = vocab_size + 1
        self.writer_stop_token_id = vocab_size
        self.writer_max_positions = max_word_bytes + 1
        self.writer_noise_warmup_steps = writer_noise_warmup_steps
        self.writer_noise_clean_ratio = writer_noise_clean_ratio
        self.writer_noise_easy_ratio = writer_noise_easy_ratio
        self.writer_noise_mid_ratio = writer_noise_mid_ratio
        self.writer_noise_hard_ratio = writer_noise_hard_ratio
        self.writer_noise_easy_min_cos = writer_noise_easy_min_cos
        self.writer_noise_easy_max_cos = writer_noise_easy_max_cos
        self.writer_noise_mid_min_cos = writer_noise_mid_min_cos
        self.writer_noise_mid_max_cos = writer_noise_mid_max_cos
        self.writer_noise_hard_min_cos = writer_noise_hard_min_cos
        self.writer_noise_hard_max_cos = writer_noise_hard_max_cos
        self.decoder_start_token_id = eos_token_id if decoder_start_token_id is None else decoder_start_token_id
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
            decoder_start_token_id=self.decoder_start_token_id,
            **kwargs,
        )
