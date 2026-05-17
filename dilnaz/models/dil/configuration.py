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
        semantic_latent_size=480,
        surface_latent_size=32,
        encoder_context_layers=2,
        max_sequence_units=4096,
        max_surface_pieces_per_unit=256,
        surface_bucket_sizes=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
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
        decoder_start_token_id=None,
        tokenizer_vocab_file="hybrid_surface_vocab.json",
        nllb_model_name="facebook/nllb-200-distilled-600M",
        nllb_src_lang="tur_Latn",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mlp_bias=False,
        checkpoint_format_version=31,
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
        if context_radius < 0:
            raise ValueError("context_radius must be >= 0")
        if num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be > 0")
        if num_encoder_layers % 2 != 0:
            raise ValueError("num_encoder_layers must be even")
        if encoder_context_layers <= 0:
            raise ValueError("encoder_context_layers must be > 0")
        if semantic_latent_size <= 0 or surface_latent_size <= 0:
            raise ValueError("semantic_latent_size and surface_latent_size must be > 0")
        if latent_size != semantic_latent_size + surface_latent_size:
            raise ValueError("latent_size must equal semantic_latent_size + surface_latent_size")
        if checkpoint_format_version != 31:
            raise ValueError("DIL factorized latent v2 requires checkpoint_format_version=31")

        self.byte_vocab_size = byte_vocab_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.encoder_context_layers = encoder_context_layers
        self.latent_size = latent_size
        self.semantic_latent_size = semantic_latent_size
        self.surface_latent_size = surface_latent_size
        self.max_surface_pieces_per_unit = max_surface_pieces_per_unit
        self.max_sequence_units = max_sequence_units
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
