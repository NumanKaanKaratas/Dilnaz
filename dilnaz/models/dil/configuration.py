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
        latent_size=512,
        max_surface_pieces_per_unit=256,
        surface_bucket_sizes=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
        byte_conv_layers=2,
        byte_conv_kernel_size=5,
        byte_conv_expansion=2,
        encoder_context_layers=8,
        encoder_layer_pattern=("sliding", "sliding", "global", "sliding", "sliding", "global", "sliding", "global"),
        encoder_attention_heads=8,
        encoder_key_value_heads=2,
        encoder_head_dim=None,
        encoder_intermediate_size=None,
        encoder_attention_window=128,
        encoder_attention_dropout=0.0,
        encoder_partial_rotary_factor=0.5,
        encoder_rope_theta=10000.0,
        encoder_gradient_checkpointing=False,
        max_sequence_units=1024,
        dil_dropout=0.10,
        distillation_weight=16.0,
        mean_geometry_weight=8.0,
        variance_weight=0.05,
        writer_loss_weight=1.0,
        writer_num_layers=6,
        writer_conv_kernel_size=5,
        writer_conv_expansion=4,
        writer_dropout=0.1,
        writer_max_window_size=32,
        writer_word_mixer_layers=2,
        writer_word_attention_heads=8,
        writer_sliding_window_size=32,
        writer_left_frozen=8,
        writer_active_size=20,
        writer_right_guard=4,
        writer_stride=20,
        writer_right_guard_loss_weight=0.2,
        writer_left_consistency_weight=0.5,
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
        writer_max_position_age=32,
        writer_use_zone_noise=True,
        writer_gradient_checkpointing=False,
        writer_future_noise_min_cos=0.970,
        writer_future_noise_max_cos=0.995,
        writer_future_noised_start_step=2000,
        writer_future_predicted_start_step=10000,
        writer_future_mixed_start_step=14000,
        writer_future_mix_ratio=0.50,
        writer_future_latent_mode="curriculum",
        decoder_start_token_id=None,
        tokenizer_vocab_file="hybrid_surface_vocab.json",
        nllb_model_name="facebook/nllb-200-distilled-600M",
        nllb_src_lang="tur_Latn",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mlp_bias=False,
        checkpoint_format_version=28,
        **kwargs,
    ):
        unsupported_encoder_keys = {"context_left_radius", "context_radius", "context_size", "target_index", "num_encoder_layers"}
        unsupported_encoder_keys = unsupported_encoder_keys.intersection(kwargs)
        if unsupported_encoder_keys:
            raise ValueError(f"legacy encoder config fields are not supported: {sorted(unsupported_encoder_keys)}")
        legacy_writer_keys = {
            "writer_output_buckets",
            "writer_initial_output_bucket",
            "writer_commit_loss_weight",
            "writer_self_conditioning_start",
            "writer_self_conditioning_final",
            "writer_refinement_steps",
            "writer_use_step_embedding",
            "writer_commit_temperature",
            "writer_commit_threshold",
            "writer_commit_min_precision",
            "writer_diffusion_steps",
            "writer_diffusion_min_mask_ratio",
            "writer_diffusion_max_mask_ratio",
            "writer_state_corruption_max_ratio",
        }
        unsupported = legacy_writer_keys.intersection(kwargs)
        if unsupported:
            raise ValueError(f"legacy writer config fields are not supported: {sorted(unsupported)}")
        if ("max_" + "word_bytes") in kwargs or ("writer_max_" + "positions") in kwargs:
            raise ValueError("fixed-width surface config is not supported; use packed variable surface fields")
        if max_surface_pieces_per_unit <= 0:
            raise ValueError("max_surface_pieces_per_unit must be > 0")
        self.surface_bucket_sizes = tuple(int(bucket) for bucket in surface_bucket_sizes)
        if not self.surface_bucket_sizes or any(bucket <= 0 for bucket in self.surface_bucket_sizes):
            raise ValueError("surface_bucket_sizes must be a non-empty positive tuple")
        if tuple(sorted(self.surface_bucket_sizes)) != self.surface_bucket_sizes:
            raise ValueError("surface_bucket_sizes must be sorted ascending")
        if self.surface_bucket_sizes[-1] < max_surface_pieces_per_unit + 1:
            raise ValueError("largest surface_bucket_sizes entry must cover max_surface_pieces_per_unit + STOP")
        if byte_conv_layers < 0 or writer_num_layers < 0:
            raise ValueError("conv layer counts must be >= 0")
        if byte_conv_kernel_size <= 0 or byte_conv_kernel_size % 2 == 0:
            raise ValueError("byte_conv_kernel_size must be a positive odd integer")
        if encoder_context_layers <= 0:
            raise ValueError("encoder_context_layers must be > 0")
        if len(tuple(encoder_layer_pattern)) != encoder_context_layers:
            raise ValueError("encoder_layer_pattern length must equal encoder_context_layers")
        if any(layer_type not in {"sliding", "global"} for layer_type in tuple(encoder_layer_pattern)):
            raise ValueError("encoder_layer_pattern entries must be sliding or global")
        if encoder_attention_heads <= 0 or encoder_key_value_heads <= 0:
            raise ValueError("encoder attention head counts must be > 0")
        if encoder_attention_heads % encoder_key_value_heads != 0:
            raise ValueError("encoder_attention_heads must be divisible by encoder_key_value_heads")
        if hidden_size % encoder_attention_heads != 0 and encoder_head_dim is None:
            raise ValueError("hidden_size must be divisible by encoder_attention_heads when encoder_head_dim is not set")
        encoder_head_dim = hidden_size // encoder_attention_heads if encoder_head_dim is None else int(encoder_head_dim)
        encoder_intermediate_size = intermediate_size if encoder_intermediate_size is None else int(encoder_intermediate_size)
        if encoder_head_dim <= 0 or encoder_intermediate_size <= 0:
            raise ValueError("encoder head/intermediate sizes must be > 0")
        if encoder_attention_window <= 0:
            raise ValueError("encoder_attention_window must be > 0")
        if not (0.0 <= encoder_attention_dropout < 1.0):
            raise ValueError("encoder_attention_dropout must be in [0, 1)")
        if encoder_partial_rotary_factor <= 0.0 or encoder_partial_rotary_factor > 1.0:
            raise ValueError("encoder_partial_rotary_factor must be in (0, 1]")
        if encoder_rope_theta <= 0.0:
            raise ValueError("encoder_rope_theta must be > 0")
        if max_sequence_units <= 0:
            raise ValueError("max_sequence_units must be > 0")
        if writer_conv_kernel_size <= 0 or writer_conv_kernel_size % 2 == 0:
            raise ValueError("writer_conv_kernel_size must be a positive odd integer")
        if writer_max_window_size <= 0:
            raise ValueError("writer_max_window_size must be > 0")
        if writer_word_mixer_layers < 0:
            raise ValueError("writer_word_mixer_layers must be >= 0")
        if writer_word_attention_heads <= 0:
            raise ValueError("writer_word_attention_heads must be > 0")
        if hidden_size % writer_word_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by writer_word_attention_heads")
        if writer_sliding_window_size <= 0:
            raise ValueError("writer_sliding_window_size must be > 0")
        if writer_left_frozen < 0 or writer_active_size <= 0 or writer_right_guard < 0:
            raise ValueError("writer window zones must satisfy left >= 0, active > 0, right >= 0")
        if writer_left_frozen + writer_active_size + writer_right_guard != writer_sliding_window_size:
            raise ValueError("writer window zones must sum to writer_sliding_window_size")
        if writer_stride <= 0 or writer_stride > writer_active_size:
            raise ValueError("writer_stride must be in 1..writer_active_size")
        if writer_sliding_window_size > writer_max_window_size:
            raise ValueError("writer_sliding_window_size must be <= writer_max_window_size")
        if writer_right_guard_loss_weight < 0.0:
            raise ValueError("writer_right_guard_loss_weight must be >= 0")
        if writer_left_consistency_weight < 0.0:
            raise ValueError("writer_left_consistency_weight must be >= 0")
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
        if writer_max_position_age <= 0:
            raise ValueError("writer_max_position_age must be > 0")
        if writer_future_noise_min_cos <= 0.0 or writer_future_noise_max_cos > 1.0 or writer_future_noise_min_cos > writer_future_noise_max_cos:
            raise ValueError("writer future noise cosine range must satisfy 0 < min <= max <= 1")
        if writer_future_noised_start_step < 0 or writer_future_predicted_start_step < 0 or writer_future_mixed_start_step < 0:
            raise ValueError("writer future curriculum start steps must be >= 0")
        if writer_future_predicted_start_step > writer_future_mixed_start_step:
            raise ValueError("writer_future_predicted_start_step must be <= writer_future_mixed_start_step")
        if not (0.0 <= writer_future_mix_ratio <= 1.0):
            raise ValueError("writer_future_mix_ratio must be in [0, 1]")
        if writer_future_latent_mode not in {"curriculum", "true", "noised", "predicted", "mixed"}:
            raise ValueError("writer_future_latent_mode must be one of curriculum,true,noised,predicted,mixed")
        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.latent_size = latent_size
        self.max_surface_pieces_per_unit = max_surface_pieces_per_unit
        self.byte_conv_layers = byte_conv_layers
        self.byte_conv_kernel_size = byte_conv_kernel_size
        self.byte_conv_expansion = byte_conv_expansion
        self.encoder_context_layers = encoder_context_layers
        self.encoder_layer_pattern = tuple(encoder_layer_pattern)
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_key_value_heads = encoder_key_value_heads
        self.encoder_head_dim = encoder_head_dim
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_attention_window = encoder_attention_window
        self.encoder_attention_dropout = encoder_attention_dropout
        self.encoder_partial_rotary_factor = encoder_partial_rotary_factor
        self.encoder_rope_theta = encoder_rope_theta
        self.encoder_gradient_checkpointing = bool(encoder_gradient_checkpointing)
        self.max_sequence_units = max_sequence_units
        self.dil_dropout = dil_dropout
        self.distillation_weight = distillation_weight
        self.mean_geometry_weight = mean_geometry_weight
        self.variance_weight = variance_weight
        self.writer_loss_weight = writer_loss_weight
        self.writer_num_layers = writer_num_layers
        self.writer_conv_kernel_size = writer_conv_kernel_size
        self.writer_conv_expansion = writer_conv_expansion
        self.writer_dropout = writer_dropout
        self.writer_vocab_size = vocab_size + 1
        self.writer_stop_token_id = vocab_size
        self.writer_bos_token_id = self.writer_vocab_size
        self.writer_empty_token_id = self.writer_vocab_size + 1
        self.writer_input_vocab_size = self.writer_vocab_size + 2
        self.writer_max_window_size = writer_max_window_size
        self.writer_word_mixer_layers = writer_word_mixer_layers
        self.writer_word_attention_heads = writer_word_attention_heads
        self.writer_sliding_window_size = writer_sliding_window_size
        self.writer_left_frozen = writer_left_frozen
        self.writer_active_size = writer_active_size
        self.writer_right_guard = writer_right_guard
        self.writer_stride = writer_stride
        self.writer_right_guard_loss_weight = writer_right_guard_loss_weight
        self.writer_left_consistency_weight = writer_left_consistency_weight
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
        self.writer_max_position_age = writer_max_position_age
        self.writer_use_zone_noise = bool(writer_use_zone_noise)
        self.writer_gradient_checkpointing = bool(writer_gradient_checkpointing)
        self.writer_future_noise_min_cos = writer_future_noise_min_cos
        self.writer_future_noise_max_cos = writer_future_noise_max_cos
        self.writer_future_noised_start_step = writer_future_noised_start_step
        self.writer_future_predicted_start_step = writer_future_predicted_start_step
        self.writer_future_mixed_start_step = writer_future_mixed_start_step
        self.writer_future_mix_ratio = writer_future_mix_ratio
        self.writer_future_latent_mode = writer_future_latent_mode
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
