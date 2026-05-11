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
        num_encoder_layers=6,
        latent_size=512,
        max_word_bytes=32,
        context_radius=2,
        byte_conv_layers=2,
        byte_conv_kernel_size=5,
        byte_conv_expansion=2,
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
        writer_commit_loss_weight=0.25,
        writer_self_conditioning_start=0.2,
        writer_self_conditioning_final=0.6,
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
        writer_refinement_steps=1,
        writer_use_step_embedding=True,
        writer_max_position_age=32,
        writer_use_zone_noise=True,
        writer_gradient_checkpointing=False,
        writer_commit_temperature=1.0,
        writer_commit_threshold=0.5,
        writer_commit_min_precision=0.98,
        writer_diffusion_steps=4,
        writer_diffusion_min_mask_ratio=0.05,
        writer_diffusion_max_mask_ratio=0.95,
        writer_state_corruption_max_ratio=0.35,
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
        checkpoint_format_version=24,
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
        if writer_commit_loss_weight < 0.0:
            raise ValueError("writer_commit_loss_weight must be >= 0")
        if not (0.0 <= writer_self_conditioning_start <= 1.0):
            raise ValueError("writer_self_conditioning_start must be in [0, 1]")
        if not (0.0 <= writer_self_conditioning_final <= 1.0):
            raise ValueError("writer_self_conditioning_final must be in [0, 1]")
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
        if writer_refinement_steps <= 0:
            raise ValueError("writer_refinement_steps must be > 0")
        if writer_max_position_age <= 0:
            raise ValueError("writer_max_position_age must be > 0")
        if writer_commit_temperature <= 0.0:
            raise ValueError("writer_commit_temperature must be > 0")
        if not (0.0 <= writer_commit_threshold <= 1.0):
            raise ValueError("writer_commit_threshold must be in [0, 1]")
        if not (0.0 < writer_commit_min_precision <= 1.0):
            raise ValueError("writer_commit_min_precision must be in (0, 1]")
        if writer_diffusion_steps <= 0:
            raise ValueError("writer_diffusion_steps must be > 0")
        if not (0.0 <= writer_diffusion_min_mask_ratio <= writer_diffusion_max_mask_ratio <= 1.0):
            raise ValueError("writer diffusion mask ratio bounds must satisfy 0 <= min <= max <= 1")
        if not (0.0 <= writer_state_corruption_max_ratio <= 1.0):
            raise ValueError("writer_state_corruption_max_ratio must be in [0, 1]")
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
        self.writer_state_vocab_size = self.writer_vocab_size + 1
        self.writer_empty_token_id = self.writer_vocab_size
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
        self.writer_commit_loss_weight = writer_commit_loss_weight
        self.writer_self_conditioning_start = writer_self_conditioning_start
        self.writer_self_conditioning_final = writer_self_conditioning_final
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
        self.writer_refinement_steps = writer_refinement_steps
        self.writer_use_step_embedding = bool(writer_use_step_embedding)
        self.writer_max_position_age = writer_max_position_age
        self.writer_use_zone_noise = bool(writer_use_zone_noise)
        self.writer_gradient_checkpointing = bool(writer_gradient_checkpointing)
        self.writer_commit_temperature = writer_commit_temperature
        self.writer_commit_threshold = writer_commit_threshold
        self.writer_commit_min_precision = writer_commit_min_precision
        self.writer_diffusion_steps = writer_diffusion_steps
        self.writer_diffusion_min_mask_ratio = writer_diffusion_min_mask_ratio
        self.writer_diffusion_max_mask_ratio = writer_diffusion_max_mask_ratio
        self.writer_state_corruption_max_ratio = writer_state_corruption_max_ratio
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
