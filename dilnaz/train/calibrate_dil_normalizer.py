import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (  # noqa: E402
    COMPILE_MODE_CHOICES,
    compile_forward,
    effective_compile_mode,
    validate_compile_environment,
)
from dil_data import (  # noqa: E402
    HybridDilBatchDataset,
    ReadyParquetDilBatchDataset,
    load_hybrid_tokenizer,
    validate_dilnaz_ready_parquet,
)
from models.configuration_dil import DilConfig  # noqa: E402
from models.modeling_dil import Dil  # noqa: E402
from train_dil import CHECKPOINT_FORMAT_VERSION, calibrate_semantic_normalizer, is_parquet_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-read-chars", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(checkpoint_dir: Path, device: torch.device):
    config = DilConfig.from_pretrained(checkpoint_dir)
    if config.checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"DIL checkpoint config format_version={config.checkpoint_format_version}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}. Retrain DIL with the current architecture."
        )
    model = Dil(config).to(device)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"DIL checkpoint format_version={checkpoint.get('format_version')}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}. Retrain DIL with the current architecture."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


def build_dataset(args, config: DilConfig, tokenizer, tokenizer_vocab_path: Path):
    if is_parquet_path(args.train_file):
        validate_dilnaz_ready_parquet(args.train_file, config, tokenizer_vocab_path)
        return ReadyParquetDilBatchDataset(
            args.train_file,
            config,
            batch_size=args.batch_size,
            repeat=False,
            max_samples=args.max_samples,
        )
    return HybridDilBatchDataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        read_chars=args.text_read_chars,
        repeat=False,
        max_samples=args.max_samples,
    )


def save_calibrated_checkpoint(checkpoint_dir: Path, model: Dil, config: DilConfig, checkpoint: dict):
    config.save_pretrained(checkpoint_dir)
    checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION
    checkpoint["model_state_dict"] = model.state_dict()
    training_state = checkpoint.setdefault("training_state", {})
    training_state["format_version"] = CHECKPOINT_FORMAT_VERSION
    training_state["semantic_normalizer_fitted"] = True
    tmp_path = checkpoint_dir / "checkpoint.pt.tmp"
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(checkpoint_dir / "checkpoint.pt")
    training_state_path = checkpoint_dir / "training_state.json"
    if training_state:
        with training_state_path.open("w", encoding="utf-8") as handle:
            json.dump(training_state, handle, indent=2)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)

    checkpoint_dir = args.checkpoint_dir.resolve()
    model, config, checkpoint = load_model(checkpoint_dir, device)
    tokenizer_vocab_path = checkpoint_dir / config.tokenizer_vocab_file
    tokenizer = load_hybrid_tokenizer(tokenizer_vocab_path)
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
    )
    dataset = build_dataset(args, config, tokenizer, tokenizer_vocab_path)
    print(
        f"semantic_normalizer_calibration_start=1 checkpoint_dir={checkpoint_dir} "
        f"device={device.type} compile_mode={compile_mode}",
        flush=True,
    )
    token_count = calibrate_semantic_normalizer(
        model,
        dataset,
        device,
        cuda_prefetch=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )
    save_calibrated_checkpoint(checkpoint_dir, model, config, checkpoint)
    print(f"semantic_normalizer_calibration_done=1 tokens={token_count}", flush=True)


if __name__ == "__main__":
    main()
