import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.configuration_dil import DilConfig  # noqa: E402
from models.modeling_dil import Dil  # noqa: E402


SOURCE_FORMAT_VERSION = 8
TARGET_FORMAT_VERSION = 9


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def migrated_config(source_dir: Path, context_radius: int) -> DilConfig:
    raw = load_json(source_dir / "config.json")
    raw.pop("context_left_radius", None)
    raw.pop("context_size", None)
    raw.pop("target_index", None)
    raw["context_radius"] = context_radius
    raw["checkpoint_format_version"] = TARGET_FORMAT_VERSION
    return DilConfig(**raw)


def copied_state_dict(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> tuple[dict, list[str]]:
    copied = []
    merged = dict(target_state)
    for key, source_tensor in source_state.items():
        target_tensor = target_state.get(key)
        if target_tensor is not None and target_tensor.shape == source_tensor.shape:
            merged[key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            copied.append(key)
    return merged, copied


def migrate_checkpoint(source_dir: Path, output_dir: Path, context_radius: int, map_location: str):
    source_checkpoint = torch.load(
        source_dir / "checkpoint.pt",
        map_location=map_location,
        weights_only=False,
    )
    if source_checkpoint["format_version"] != SOURCE_FORMAT_VERSION:
        raise ValueError(f"source checkpoint format_version must be {SOURCE_FORMAT_VERSION}")

    config = migrated_config(source_dir, context_radius)
    model = Dil(config)
    merged_state, copied = copied_state_dict(source_checkpoint["model_state_dict"], model.state_dict())
    model.load_state_dict(merged_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    vocab_file = source_dir / config.tokenizer_vocab_file
    if vocab_file.exists():
        shutil.copyfile(vocab_file, output_dir / config.tokenizer_vocab_file)

    source_training_state = source_checkpoint.get("training_state", {})
    training_state = {
        "format_version": TARGET_FORMAT_VERSION,
        "step": 0,
        "metrics": {},
        "migrated_from_step": int(source_training_state.get("step", 0)),
        "migrated_from": str(source_dir.resolve()),
        "copied_tensor_count": len(copied),
        "context_radius": config.context_radius,
        "target_index": config.target_index,
    }
    torch.save(
        {
            "format_version": TARGET_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "training_state": training_state,
            "rng_state": None,
        },
        output_dir / "checkpoint.pt",
    )
    write_json(output_dir / "training_state.json", training_state)
    return copied


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--map-location", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.context_radius < 0:
        raise ValueError("--context-radius must be >= 0")
    copied = migrate_checkpoint(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.context_radius,
        args.map_location,
    )
    print(f"migrated checkpoint with {len(copied)} copied tensors -> {args.output_dir}")


if __name__ == "__main__":
    main()
