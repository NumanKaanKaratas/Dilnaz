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
        max_sequence_units=4096,
        dil_dropout=0.10,
        distillation_weight=16.0,
        mean_geometry_weight=8.0,
        variance_weight=0.05,
        writer_num_layers=6,
        writer_conv_kernel_size=5,
        writer_conv_expansion=4,
        writer_dropout=0.1,
        writer_gradient_checkpointing=False,
        decoder_start_token_id=None,
        tokenizer_vocab_file="hybrid_surface_vocab.json",
        nllb_model_name="facebook/nllb-200-distilled-600M",
        nllb_src_lang="tur_Latn",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mlp_bias=False,
        checkpoint_format_version=30,
        **kwargs,
    ):
        self.surface_bucket_sizes = tuple(int(bucket) for bucket in surface_bucket_sizes)
        if not self.surface_bucket_sizes or any(bucket <= 0 for bucket in self.surface_bucket_sizes):
            raise ValueError("surface_bucket_sizes must be a non-empty positive tuple")
        if tuple(sorted(self.surface_bucket_sizes)) != self.surface_bucket_sizes:
            raise ValueError("surface_bucket_sizes must be sorted ascending")
        if self.surface_bucket_sizes[-1] < max_surface_pieces_per_unit + 1:
            raise ValueError("largest surface_bucket_sizes entry must cover max_surface_pieces_per_unit + STOP")
        if max_surface_pieces_per_unit <= 0:
            raise ValueError("max_surface_pieces_per_unit must be > 0")
        if byte_conv_layers < 0 or writer_num_layers < 0:
            raise ValueError("conv layer counts must be >= 0")
        if byte_conv_kernel_size <= 0 or byte_conv_kernel_size % 2 == 0:
            raise ValueError("byte_conv_kernel_size must be a positive odd integer")
        if writer_conv_kernel_size <= 0 or writer_conv_kernel_size % 2 == 0:
            raise ValueError("writer_conv_kernel_size must be a positive odd integer")
        if byte_conv_expansion <= 0 or writer_conv_expansion <= 0:
            raise ValueError("conv expansion values must be > 0")
        if not 0.0 <= writer_dropout < 1.0:
            raise ValueError("writer_dropout must be in [0, 1)")
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
        if not 0.0 <= encoder_attention_dropout < 1.0:
            raise ValueError("encoder_attention_dropout must be in [0, 1)")
        if encoder_partial_rotary_factor <= 0.0 or encoder_partial_rotary_factor > 1.0:
            raise ValueError("encoder_partial_rotary_factor must be in (0, 1]")
        if encoder_rope_theta <= 0.0:
            raise ValueError("encoder_rope_theta must be > 0")
        if max_sequence_units <= 0:
            raise ValueError("max_sequence_units must be > 0")

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
        self.writer_num_layers = writer_num_layers
        self.writer_conv_kernel_size = writer_conv_kernel_size
        self.writer_conv_expansion = writer_conv_expansion
        self.writer_dropout = writer_dropout
        self.writer_gradient_checkpointing = bool(writer_gradient_checkpointing)
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
