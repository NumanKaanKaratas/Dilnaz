import argparse
import json
import queue
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cuda_sync,
    effective_compile_mode,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dil_data import (
    HybridDilBatchDataset,
    NllbTeacher,
    ReadyParquetDilBatchDataset,
    ResidentDilBatcher,
    ResidentDilEvalLoader,
    ResidentReadyParquetBatcher,
    validate_dilnaz_ready_parquet,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
)
from dilnaz_config import DIL_MODEL_DEFAULTS, DIL_TRAIN_DEFAULTS
from models.configuration_dil import DilConfig
from models.modeling_dil import Dil
from tokenization import default_vocab_path


CHECKPOINT_FORMAT_VERSION = 9
DATALOADER_WORKER_EXIT = "DataLoader worker"


class AsyncTeacherBatchSource:
    def __init__(self, train_iter, teacher, device: torch.device, max_batch_reuse: int):
        self.train_iter = train_iter
        self.teacher = teacher
        self.device = device
        self.max_batch_reuse = max_batch_reuse
        self.ready: queue.Queue[dict | BaseException] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.produce, daemon=True)
        self.worker.start()

    def produce(self):
        try:
            while not self.stop_event.is_set():
                batch = next(self.train_iter)
                teacher_layers, teacher_mask = self.teacher.teacher_layers(batch)
                batch["teacher_layers"] = teacher_layers
                batch["teacher_mask"] = teacher_mask
                cuda_sync(self.device)
                self.ready.put(batch)
        except BaseException as error:
            self.ready.put(error)

    def unwrap(self, item: dict | BaseException) -> dict:
        if isinstance(item, BaseException):
            raise item
        return item

    def first(self) -> tuple[dict, int, float]:
        start = time.perf_counter()
        return self.unwrap(self.ready.get()), 1, time.perf_counter() - start

    def next_after_step(self, current: dict, seen_count: int) -> tuple[dict, int, float]:
        try:
            return self.unwrap(self.ready.get_nowait()), 1, 0.0
        except queue.Empty:
            if seen_count < self.max_batch_reuse:
                return current, seen_count + 1, 0.0
            start = time.perf_counter()
            return self.unwrap(self.ready.get()), 1, time.perf_counter() - start

    def close(self):
        self.stop_event.set()


def make_scheduler(optimizer, learning_rate: float, warmup_steps: int):
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def json_training_state(config, step: int, metrics: dict, compile_mode: str):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "metrics": metrics,
        "compile_mode": compile_mode,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "max_word_bytes": config.max_word_bytes,
        "context_radius": config.context_radius,
        "target_index": config.target_index,
        "latent_size": config.latent_size,
        "ce_weight": config.ce_weight,
        "distillation_weight": config.distillation_weight,
    }


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    config,
    tokenizer_vocab_path: Path,
    step: int,
    metrics: dict,
    compile_mode: str,
    checkpoint_name: str = "",
):
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    dst_vocab = checkpoint_dir / config.tokenizer_vocab_file
    if tokenizer_vocab_path.resolve() != dst_vocab.resolve():
        shutil.copyfile(tokenizer_vocab_path, dst_vocab)
    state = json_training_state(config, step, metrics, compile_mode)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": state,
            "rng_state": rng_state(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return checkpoint_dir


def restore_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, dict]:
    checkpoint = load_checkpoint(path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    rng = checkpoint.get("rng_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if rng is not None:
        restore_rng_state(rng)
    training_state = checkpoint["training_state"]
    return int(training_state["step"]), dict(training_state["metrics"])


def is_dataloader_worker_exit(error: RuntimeError) -> bool:
    message = str(error)
    return DATALOADER_WORKER_EXIT in message and "exited unexpectedly" in message


def model_inputs(batch: dict) -> dict:
    return {
        "input_ids": batch["input_ids"],
        "word_masks": batch["word_masks"],
        "labels": batch["labels"],
        "length_labels": batch["length_labels"],
        "teacher_layers": batch["teacher_layers"],
        "teacher_mask": batch["teacher_mask"],
    }


@torch.no_grad()
def evaluate(model, eval_loader, teacher, device, autocast_enabled: bool, cuda_prefetch: bool, max_batches: int):
    model.eval()
    total = {
        "loss": 0.0,
        "ce": 0.0,
        "ce_weighted": 0.0,
        "len": 0.0,
        "kl": 0.0,
        "geom_l1": 0.0,
        "geom_l2": 0.0,
        "geom_l3": 0.0,
        "geom_l4": 0.0,
        "geom_mean": 0.0,
        "var": 0.0,
        "byte_acc": 0.0,
        "len_acc": 0.0,
        "batches": 0,
    }
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        if "teacher_layers" not in batch:
            if teacher is None:
                raise ValueError("eval batch has no teacher_layers and no NLLB teacher is available")
            teacher_layers, teacher_mask = teacher.teacher_layers(batch)
            batch["teacher_layers"] = teacher_layers
            batch["teacher_mask"] = teacher_mask
        with autocast_context(autocast_enabled):
            outputs = model(**model_inputs(batch))
        total["loss"] += float(outputs.loss.detach().cpu())
        total["ce"] += float(outputs.ce_loss.detach().cpu())
        total["ce_weighted"] += float((outputs.ce_loss * model.config.ce_weight).detach().cpu())
        total["len"] += float(outputs.length_loss.detach().cpu())
        total["kl"] += float(outputs.kl_loss.detach().cpu())
        layer_losses = outputs.layer_geometry_losses.detach().cpu().tolist()
        for idx in range(4):
            total[f"geom_l{idx + 1}"] += float(layer_losses[idx]) if idx < len(layer_losses) else 0.0
        total["geom_mean"] += float(outputs.mean_geometry_loss.detach().cpu())
        total["var"] += float(outputs.variance_loss.detach().cpu())
        total["byte_acc"] += float(outputs.byte_acc.detach().cpu())
        total["len_acc"] += float(outputs.length_acc.detach().cpu())
        total["batches"] += 1
        if batch_idx >= max_batches:
            break

    model.train()
    batches = max(total.pop("batches"), 1)
    return {f"eval_{key}": value / batches for key, value in total.items()}


def format_log(step, metrics):
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"ce={metrics['ce']:.4f}",
        f"ce_w={metrics['ce_weighted']:.4f}",
        f"len={metrics['len']:.4f}",
        f"kl={metrics['kl']:.2f}",
        f"kl_w={metrics['kl_weighted']:.4f}",
        f"len_w={metrics['length_weighted']:.4f}",
        f"geom_l1={metrics['geom_l1']:.4f}",
        f"geom_l2={metrics['geom_l2']:.4f}",
        f"geom_l3={metrics['geom_l3']:.4f}",
        f"geom_l4={metrics['geom_l4']:.4f}",
        f"geom_mean={metrics['geom_mean']:.4f}",
        f"var={metrics['var']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"len_acc={metrics['len_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"w/s={metrics['windows_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    if "source_lines_seen" in metrics:
        fields.append(f"total/row={int(metrics['source_lines_seen'])}")
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer-vocab", type=Path, default=default_vocab_path())
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument(
        "--data-mode",
        choices=("streaming", "resident"),
        default=DIL_TRAIN_DEFAULTS["data_mode"],
    )
    parser.add_argument("--max-steps", type=int, default=DIL_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DIL_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--nllb-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["nllb_batch_size"])
    parser.add_argument("--max-batch-reuse", type=int, default=DIL_TRAIN_DEFAULTS["max_batch_reuse"])
    parser.add_argument("--text-read-chars", type=int, default=DIL_TRAIN_DEFAULTS["text_read_chars"])
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DIL_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DIL_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=DIL_TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--max-grad-norm", type=float, default=DIL_TRAIN_DEFAULTS["max_grad_norm"])
    parser.add_argument("--log-every", type=int, default=DIL_TRAIN_DEFAULTS["log_every"])
    parser.add_argument("--checkpoint-every", type=int, default=DIL_TRAIN_DEFAULTS["checkpoint_every"])
    parser.add_argument("--eval-every", type=int, default=DIL_TRAIN_DEFAULTS["eval_every"])
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=DIL_TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DIL_TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-samples", type=int, default=DIL_TRAIN_DEFAULTS["max_samples"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=DIL_MODEL_DEFAULTS["hidden_size"])
    parser.add_argument("--intermediate-size", type=int, default=DIL_MODEL_DEFAULTS["intermediate_size"])
    parser.add_argument("--num-encoder-layers", type=int, default=DIL_MODEL_DEFAULTS["num_encoder_layers"])
    parser.add_argument("--num-decoder-layers", type=int, default=DIL_MODEL_DEFAULTS["num_decoder_layers"])
    parser.add_argument("--latent-size", type=int, default=DIL_MODEL_DEFAULTS["latent_size"])
    parser.add_argument("--max-word-bytes", type=int, default=DIL_MODEL_DEFAULTS["max_word_bytes"])
    parser.add_argument("--context-radius", type=int, default=DIL_MODEL_DEFAULTS["context_radius"])
    parser.add_argument("--dil-dropout", type=float, default=DIL_MODEL_DEFAULTS["dil_dropout"])
    parser.add_argument("--kl-clamp", type=float, default=DIL_MODEL_DEFAULTS["kl_clamp"])
    parser.add_argument("--kl-weight", type=float, default=DIL_MODEL_DEFAULTS["kl_weight"])
    parser.add_argument("--ce-weight", type=float, default=DIL_MODEL_DEFAULTS["ce_weight"])
    parser.add_argument("--distillation-weight", type=float, default=DIL_MODEL_DEFAULTS["distillation_weight"])
    parser.add_argument("--layer-geometry-weight", type=float, default=DIL_MODEL_DEFAULTS["layer_geometry_weight"])
    parser.add_argument("--mean-geometry-weight", type=float, default=DIL_MODEL_DEFAULTS["mean_geometry_weight"])
    parser.add_argument("--variance-weight", type=float, default=DIL_MODEL_DEFAULTS["variance_weight"])
    parser.add_argument("--length-loss-weight", type=float, default=DIL_MODEL_DEFAULTS["length_loss_weight"])
    parser.add_argument("--nllb-model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--nllb-src-lang", default="tur_Latn")
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.nllb_batch_size <= 0:
        raise ValueError("--nllb-batch-size must be > 0")
    if args.max_batch_reuse <= 0:
        raise ValueError("--max-batch-reuse must be > 0")
    if args.ce_weight <= 0:
        raise ValueError("--ce-weight must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.context_radius < 0:
        raise ValueError("--context-radius must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.data_mode == "resident" and args.max_samples > 0:
        raise ValueError("--max-samples is not supported with --data-mode resident")


def is_parquet_path(path: Path | None) -> bool:
    return path is not None and path.suffix.casefold() == ".parquet"


def build_config(args, tokenizer):
    if args.resume is not None:
        return DilConfig.from_pretrained(args.resume.parent)
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        latent_size=args.latent_size,
        max_word_bytes=args.max_word_bytes,
        context_radius=args.context_radius,
        dil_dropout=args.dil_dropout,
        kl_clamp=args.kl_clamp,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        distillation_weight=args.distillation_weight,
        layer_geometry_weight=args.layer_geometry_weight,
        mean_geometry_weight=args.mean_geometry_weight,
        variance_weight=args.variance_weight,
        length_loss_weight=args.length_loss_weight,
        tokenizer_vocab_file=args.tokenizer_vocab.name,
        nllb_model_name=args.nllb_model_name,
        nllb_src_lang=args.nllb_src_lang,
    )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer_vocab_path = args.tokenizer_vocab
    if args.resume is not None:
        resume_config = DilConfig.from_pretrained(args.resume.parent)
        tokenizer_vocab_path = args.resume.parent / resume_config.tokenizer_vocab_file
    tokenizer = load_hybrid_tokenizer(tokenizer_vocab_path)
    config = build_config(args, tokenizer)
    train_is_parquet = is_parquet_path(args.train_file)
    eval_is_parquet = is_parquet_path(args.eval_file)
    if train_is_parquet:
        validate_dilnaz_ready_parquet(args.train_file, config, tokenizer_vocab_path)
    if eval_is_parquet:
        validate_dilnaz_ready_parquet(args.eval_file, config, tokenizer_vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    teacher_dtype = torch.bfloat16 if autocast_enabled else torch.float32
    cuda_prefetch = bool(device.type == "cuda" and not args.no_cuda_prefetch)

    base_model = Dil(config).to(device)
    base_model.train()
    base_model.set_compiled_forwards(
        encoder_forward=compile_forward(base_model.encoder.forward, compile_mode, "DilEncoderCore"),
        decode_forward=compile_forward(base_model._decode_from_latents_impl, compile_mode, "DilDecoderRenderer"),
    )
    model = base_model
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps)
    needs_online_teacher = not train_is_parquet or (args.eval_file is not None and not eval_is_parquet)
    teacher = None
    if needs_online_teacher:
        teacher = NllbTeacher(
            config.nllb_model_name,
            config.nllb_src_lang,
            device,
            teacher_dtype,
            batch_size=args.nllb_batch_size,
        )

    start_step = 0
    last_metrics = {}
    if args.resume is not None:
        start_step, last_metrics = restore_checkpoint(args.resume, model, optimizer, scheduler, device)

    print(
        f"device={device.type} bf16={int(autocast_enabled)} compile_mode={compile_mode} "
        f"data_mode={args.data_mode} resume_step={start_step} "
        f"teacher_source={'parquet nllb=disabled' if train_is_parquet else 'online_nllb'} "
        f"vocab_size={config.vocab_size} latent_size={config.latent_size} hidden_size={config.hidden_size}",
        flush=True,
    )

    if train_is_parquet:
        train_dataset = ReadyParquetDilBatchDataset(
            args.train_file,
            config,
            batch_size=args.batch_size,
            repeat=True,
            max_samples=args.max_samples,
        )
    else:
        assert teacher is not None
        train_dataset = HybridDilBatchDataset(
            args.train_file,
            config,
            tokenizer,
            batch_size=args.batch_size,
            read_chars=args.text_read_chars,
            repeat=True,
            max_samples=args.max_samples,
            teacher_tokenizer=teacher.tokenizer,
            teacher_max_tokens=teacher.max_encoder_tokens,
        )
    eval_dataset = None
    if args.eval_every > 0:
        if eval_is_parquet:
            eval_dataset = ReadyParquetDilBatchDataset(
                args.eval_file,
                config,
                batch_size=args.eval_batch_size,
                repeat=False,
            )
        else:
            assert teacher is not None
            eval_dataset = HybridDilBatchDataset(
                args.eval_file,
                config,
                tokenizer,
                batch_size=args.eval_batch_size,
                read_chars=args.text_read_chars,
                repeat=False,
                teacher_tokenizer=teacher.tokenizer,
                teacher_max_tokens=teacher.max_encoder_tokens,
            )

    if args.data_mode == "resident":
        print("resident_data_prepare_start=1", flush=True)
        if train_is_parquet:
            train_iter = ResidentReadyParquetBatcher.from_dataset(
                train_dataset,
                args.batch_size,
                device,
                args.seed + start_step,
            )
        else:
            assert teacher is not None
            train_iter = ResidentDilBatcher.from_dataset(
                train_dataset,
                teacher,
                args.batch_size,
                device,
                args.seed + start_step,
            )
        print(f"resident_data_prepare_done=1 batches={len(train_iter.batches)}", flush=True)
        eval_loader = None
        if eval_dataset is not None:
            print("resident_eval_prepare_start=1", flush=True)
            if eval_is_parquet:
                eval_loader = ResidentDilEvalLoader(
                    ResidentReadyParquetBatcher.from_dataset(
                        eval_dataset,
                        args.eval_batch_size,
                        device,
                        args.seed + 1,
                    )
                )
            else:
                assert teacher is not None
                eval_loader = ResidentDilEvalLoader(
                    ResidentDilBatcher.from_dataset(
                        eval_dataset,
                        teacher,
                        args.eval_batch_size,
                        device,
                        args.seed + 1,
                    )
                )
            print(f"resident_eval_prepare_done=1 batches={len(eval_loader.batches)}", flush=True)
        batch_source = None
    else:
        train_loader = make_dil_batch_loader(
            train_dataset,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = make_dil_batch_loader(
                eval_dataset,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=args.prefetch_factor,
            )
        train_iter = DeviceBatchPrefetcher(train_loader, device, cuda_prefetch)
        if train_is_parquet:
            batch_source = None
        else:
            assert teacher is not None
            batch_source = AsyncTeacherBatchSource(train_iter, teacher, device, args.max_batch_reuse)

    log_start = time.perf_counter()
    log_tokens = 0
    log_windows = 0
    log_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    source_lines_seen: set[int] = set()
    metric_sums = {
        "loss": 0.0,
        "ce": 0.0,
        "ce_weighted": 0.0,
        "len": 0.0,
        "kl": 0.0,
        "kl_weighted": 0.0,
        "length_weighted": 0.0,
        "geom_l1": 0.0,
        "geom_l2": 0.0,
        "geom_l3": 0.0,
        "geom_l4": 0.0,
        "geom_mean": 0.0,
        "var": 0.0,
        "byte_acc": 0.0,
        "len_acc": 0.0,
    }
    completed_step = start_step
    current_batch = None
    current_batch_seen = 0
    if batch_source is not None:
        current_batch, current_batch_seen, initial_wait = batch_source.first()
        data_seconds += initial_wait

    def save_interrupted():
        interrupted_dir = save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scheduler,
            config,
            tokenizer_vocab_path,
            completed_step,
            last_metrics,
            compile_mode,
        )
        print(f"interrupted_saved={interrupted_dir}", flush=True)

    try:
        for step in range(start_step + 1, args.max_steps + 1):
            if args.data_mode == "resident":
                batch = next(train_iter)
            elif train_is_parquet:
                data_start = time.perf_counter()
                batch = next(train_iter)
                data_seconds += time.perf_counter() - data_start
            else:
                batch = current_batch

            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_enabled):
                outputs = model(**model_inputs(batch))

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step
            if batch_source is not None:
                current_batch, current_batch_seen, wait_seconds = batch_source.next_after_step(
                    current_batch,
                    current_batch_seen,
                )
                data_seconds += wait_seconds

            real_tokens = int(batch["labels"].ne(-100).sum().detach().cpu())
            log_tokens += real_tokens
            log_windows += int(batch["labels"].shape[0])
            log_steps += 1
            if "source_line_ids" in batch:
                source_lines_seen.update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())
            metric_sums["loss"] += float(outputs.loss.detach().cpu())
            metric_sums["ce"] += float(outputs.ce_loss.detach().cpu())
            metric_sums["ce_weighted"] += float((outputs.ce_loss * config.ce_weight).detach().cpu())
            metric_sums["len"] += float(outputs.length_loss.detach().cpu())
            metric_sums["kl"] += float(outputs.kl_loss.detach().cpu())
            metric_sums["kl_weighted"] += float((outputs.kl_loss * config.kl_weight).detach().cpu())
            metric_sums["length_weighted"] += float(
                (outputs.length_loss * config.length_loss_weight).detach().cpu()
            )
            layer_losses = outputs.layer_geometry_losses.detach().cpu().tolist()
            for idx in range(4):
                metric_sums[f"geom_l{idx + 1}"] += float(layer_losses[idx]) if idx < len(layer_losses) else 0.0
            metric_sums["geom_mean"] += float(outputs.mean_geometry_loss.detach().cpu())
            metric_sums["var"] += float(outputs.variance_loss.detach().cpu())
            metric_sums["byte_acc"] += float(outputs.byte_acc.detach().cpu())
            metric_sums["len_acc"] += float(outputs.length_acc.detach().cpu())

            should_log = step % args.log_every == 0 or step == start_step + 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["windows_per_second"] = log_windows / elapsed
                averaged["steps_per_second"] = log_steps / elapsed
                if source_lines_seen:
                    averaged["source_lines_seen"] = len(source_lines_seen)
                if should_eval:
                    averaged.update(
                        evaluate(
                            model,
                            eval_loader,
                            teacher,
                            device,
                            autocast_enabled,
                            cuda_prefetch,
                            args.max_eval_batches,
                        )
                    )
                print(format_log(step, averaged), flush=True)
                last_metrics = averaged
                log_start = time.perf_counter()
                log_tokens = 0
                log_windows = 0
                log_steps = 0
                data_seconds = 0.0
                compute_seconds = 0.0
                for key in metric_sums:
                    metric_sums[key] = 0.0

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    tokenizer_vocab_path,
                    step,
                    last_metrics,
                    compile_mode,
                    checkpoint_name=f"checkpoint-{step}",
                )
    except KeyboardInterrupt:
        if batch_source is not None:
            batch_source.close()
        save_interrupted()
        return
    except RuntimeError as error:
        if batch_source is not None:
            batch_source.close()
        if not is_dataloader_worker_exit(error):
            raise
        save_interrupted()
        return

    if batch_source is not None:
        batch_source.close()

    final_dir = save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        scheduler,
        config,
        tokenizer_vocab_path,
        args.max_steps,
        last_metrics,
        compile_mode,
    )
    print(f"saved={final_dir}", flush=True)


if __name__ == "__main__":
    main()
