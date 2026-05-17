import argparse
import random
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from dilnaz.train.common.runtime import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cudagraph_step_begin,
    cuda_sync,
    effective_compile_mode,
    validate_compile_environment,
)
from dilnaz.train.common.trainer_core import make_adamw_param_groups
from dilnaz.train.data.dil_data import load_hybrid_tokenizer, make_dil_batch_loader
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS, DIL_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.dil import Dil
from dilnaz.train.data.parallel_dil_data import (
    DEFAULT_PARALLEL_NLLB_MODEL,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    ParallelDilBatchDataset,
    ParallelNllbTeacher,
    parallel_total_loss,
)
from dilnaz.tokenization import default_vocab_path
from dilnaz.train.dil.train import (
    prepare_writer_for_surface_training,
    is_dataloader_worker_exit,
    make_scheduler,
    model_inputs,
    restore_checkpoint,
    save_checkpoint,
)


class ParallelAsyncTeacherBatchSource:
    def __init__(self, train_iter, teacher: ParallelNllbTeacher, device: torch.device, max_batch_reuse: int):
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
                self.teacher.materialize(batch)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer-vocab", type=Path, default=default_vocab_path())
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--max-steps", type=int, default=DIL_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DIL_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--nllb-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["nllb_batch_size"])
    parser.add_argument("--max-batch-reuse", type=int, default=DIL_TRAIN_DEFAULTS["max_batch_reuse"])
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DIL_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DIL_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=DIL_TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DIL_TRAIN_DEFAULTS["gradient_accumulation_steps"])
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
    parser.add_argument("--latent-size", type=int, default=DIL_MODEL_DEFAULTS["latent_size"])
    parser.add_argument("--semantic-latent-size", type=int, default=DIL_MODEL_DEFAULTS["semantic_latent_size"])
    parser.add_argument("--surface-latent-size", type=int, default=DIL_MODEL_DEFAULTS["surface_latent_size"])
    parser.add_argument("--encoder-context-layers", type=int, default=DIL_MODEL_DEFAULTS["encoder_context_layers"])
    parser.add_argument("--max-surface-pieces-per-unit", type=int, default=DIL_MODEL_DEFAULTS["max_surface_pieces_per_unit"])
    parser.add_argument("--byte-conv-layers", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_layers"])
    parser.add_argument("--byte-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_kernel_size"])
    parser.add_argument("--byte-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_expansion"])
    parser.add_argument("--dil-dropout", type=float, default=DIL_MODEL_DEFAULTS["dil_dropout"])
    parser.add_argument("--distillation-weight", type=float, default=DIL_MODEL_DEFAULTS["distillation_weight"])
    parser.add_argument("--mean-geometry-weight", type=float, default=DIL_MODEL_DEFAULTS["mean_geometry_weight"])
    parser.add_argument("--variance-weight", type=float, default=DIL_MODEL_DEFAULTS["variance_weight"])
    parser.add_argument("--writer-num-layers", type=int, default=DIL_MODEL_DEFAULTS["writer_num_layers"])
    parser.add_argument("--writer-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_kernel_size"])
    parser.add_argument("--writer-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_expansion"])
    parser.add_argument("--writer-dropout", type=float, default=DIL_MODEL_DEFAULTS["writer_dropout"])
    parser.add_argument("--parallel-alignment-weight", type=float, default=1.0)
    parser.add_argument("--nllb-model-name", default=DEFAULT_PARALLEL_NLLB_MODEL)
    parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--align-layer", type=int, default=-1)
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
    if args.parallel_alignment_weight < 0:
        raise ValueError("--parallel-alignment-weight must be >= 0")
    if args.byte_conv_layers < 0:
        raise ValueError("--byte-conv-layers must be >= 0")
    if args.semantic_latent_size <= 0 or args.surface_latent_size <= 0:
        raise ValueError("--semantic-latent-size and --surface-latent-size must be > 0")
    if args.latent_size != args.semantic_latent_size + args.surface_latent_size:
        raise ValueError("--latent-size must equal semantic + surface latent sizes")
    if args.encoder_context_layers <= 0:
        raise ValueError("--encoder-context-layers must be > 0")
    if args.byte_conv_kernel_size <= 0 or args.byte_conv_kernel_size % 2 == 0:
        raise ValueError("--byte-conv-kernel-size must be a positive odd integer")
    if args.byte_conv_expansion <= 0:
        raise ValueError("--byte-conv-expansion must be > 0")
    if args.writer_num_layers < 0:
        raise ValueError("--writer-num-layers must be >= 0")
    if args.writer_conv_kernel_size <= 0 or args.writer_conv_kernel_size % 2 == 0:
        raise ValueError("--writer-conv-kernel-size must be a positive odd integer")
    if args.writer_conv_expansion <= 0:
        raise ValueError("--writer-conv-expansion must be > 0")
    if not 0.0 <= args.writer_dropout < 1.0:
        raise ValueError("--writer-dropout must be inside [0, 1)")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")

def build_config(args, tokenizer):
    if args.resume is not None:
        return DilConfig.from_pretrained(args.resume.parent)
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        latent_size=args.latent_size,
        semantic_latent_size=args.semantic_latent_size,
        surface_latent_size=args.surface_latent_size,
        encoder_context_layers=args.encoder_context_layers,
        max_surface_pieces_per_unit=args.max_surface_pieces_per_unit,
        byte_conv_layers=args.byte_conv_layers,
        byte_conv_kernel_size=args.byte_conv_kernel_size,
        byte_conv_expansion=args.byte_conv_expansion,
        dil_dropout=args.dil_dropout,
        distillation_weight=args.distillation_weight,
        mean_geometry_weight=args.mean_geometry_weight,
        variance_weight=args.variance_weight,
        max_sequence_units=DIL_MODEL_DEFAULTS["max_sequence_units"],
        writer_num_layers=args.writer_num_layers,
        writer_conv_kernel_size=args.writer_conv_kernel_size,
        writer_conv_expansion=args.writer_conv_expansion,
        writer_dropout=args.writer_dropout,
        tokenizer_vocab_file=args.tokenizer_vocab.name,
        nllb_model_name=args.nllb_model_name,
        nllb_src_lang=args.source_lang,
    )


def teacher_dtype_for(device: torch.device, autocast_enabled: bool) -> torch.dtype:
    del autocast_enabled
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16


def format_parallel_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"dil={metrics['dil_loss']:.4f}",
        f"parallel={metrics['parallel']:.4f}",
        f"parallel_w={metrics['parallel_weighted']:.4f}",
        f"distill={metrics['distill']:.4f}",
        f"geom_mean={metrics['geom_mean']:.4f}",
        f"var={metrics['var']:.4f}",
        f"align_groups={metrics['align_groups']:.1f}",
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


def empty_metric_sums() -> dict[str, float]:
    return {
        "loss": 0.0,
        "dil_loss": 0.0,
        "parallel": 0.0,
        "parallel_weighted": 0.0,
        "distill": 0.0,
        "geom_mean": 0.0,
        "var": 0.0,
        "align_groups": 0.0,
    }


def accumulate_metrics(metric_sums: dict[str, float], loss, outputs, parallel_loss, batch, config, weight: float):
    metric_sums["loss"] += float(loss.detach().cpu())
    metric_sums["dil_loss"] += float(outputs.loss.detach().cpu())
    metric_sums["parallel"] += float(parallel_loss.detach().cpu())
    metric_sums["parallel_weighted"] += float((parallel_loss * weight).detach().cpu())
    metric_sums["distill"] += float(outputs.distill_loss.detach().cpu())
    metric_sums["geom_mean"] += float(outputs.mean_geometry_loss.detach().cpu())
    metric_sums["var"] += float(outputs.variance_loss.detach().cpu())
    metric_sums["align_groups"] += float(batch["parallel_alignment_scores"].shape[0])


@torch.no_grad()
def evaluate_parallel(
    model,
    eval_loader,
    teacher: ParallelNllbTeacher,
    device,
    compile_mode: str,
    autocast_enabled: bool,
    cuda_prefetch: bool,
    max_batches: int,
    parallel_alignment_weight: float,
):
    model.eval()
    total = empty_metric_sums()
    batches = 0
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        teacher.materialize(batch)
        cudagraph_step_begin(device, compile_mode)
        with autocast_context(autocast_enabled):
            outputs = model(**model_inputs(batch))
            loss, parallel_loss = parallel_total_loss(
                outputs,
                batch,
                parallel_alignment_weight,
                model.config.semantic_latent_size,
                model.config.surface_latent_size,
            )
        accumulate_metrics(total, loss, outputs, parallel_loss, batch, model.config, parallel_alignment_weight)
        batches += 1
        if batch_idx >= max_batches:
            break

    model.train()
    return {f"eval_{key}": value / max(batches, 1) for key, value in total.items()}


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    teacher_dtype = teacher_dtype_for(device, autocast_enabled)
    cuda_prefetch = bool(device.type == "cuda" and not args.no_cuda_prefetch)

    base_model = Dil(config).to(device)
    base_model.train()
    prepare_writer_for_surface_training(base_model)
    base_model.set_compiled_forwards(
        encoder_forward=compile_forward(base_model.encoder.forward, compile_mode, "DilEncoderCore"),
    )
    model = base_model
    trainable_named_parameters = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    trainable_parameters = [param for _, param in trainable_named_parameters]
    optimizer = AdamW(
        make_adamw_param_groups(trainable_named_parameters, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps, args.max_steps)
    teacher = ParallelNllbTeacher(
        args.nllb_model_name,
        args.source_lang,
        args.target_lang,
        device,
        teacher_dtype,
        batch_size=args.nllb_batch_size,
        align_layer=args.align_layer,
    )

    start_step = 0
    last_metrics = {}
    if args.resume is not None:
        start_step, last_metrics = restore_checkpoint(args.resume, model, optimizer, scheduler, device)

    print(
        f"device={device.type} bf16={int(autocast_enabled)} compile_mode={compile_mode} "
        f"resume_step={start_step} teacher_source=parallel_online_nllb "
        f"nllb={args.nllb_model_name} source_lang={args.source_lang} target_lang={args.target_lang} "
        f"teacher_dtype={str(teacher_dtype).replace('torch.', '')} nllb_batch={args.nllb_batch_size} "
        f"parallel_weight={args.parallel_alignment_weight} vocab_size={config.vocab_size} "
        f"latent_size={config.latent_size} hidden_size={config.hidden_size}",
        flush=True,
    )

    train_dataset = ParallelDilBatchDataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        repeat=True,
        max_samples=args.max_samples,
    )
    eval_dataset = None
    if args.eval_every > 0:
        eval_dataset = ParallelDilBatchDataset(
            args.eval_file,
            config,
            tokenizer,
            batch_size=args.eval_batch_size,
            repeat=False,
        )

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
    batch_source = ParallelAsyncTeacherBatchSource(train_iter, teacher, device, args.max_batch_reuse)

    log_start = time.perf_counter()
    log_tokens = 0
    log_windows = 0
    log_steps = 0
    log_micro_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    source_lines_seen: set[int] = set()
    metric_sums = empty_metric_sums()
    completed_step = start_step
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
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.gradient_accumulation_steps):
                batch = current_batch
                cudagraph_step_begin(device, compile_mode)
                model_batch = model_inputs(batch)
                model_batch["training_step"] = step
                with autocast_context(autocast_enabled):
                    outputs = model(**model_batch)
                    loss, parallel_loss = parallel_total_loss(
                        outputs,
                        batch,
                        args.parallel_alignment_weight,
                        model.config.semantic_latent_size,
                        model.config.surface_latent_size,
                    )

                (loss / args.gradient_accumulation_steps).backward()
                token_count = int(batch["surface"].mask.sum().detach().cpu())
                window_count = int(batch["surface"].unit_mask.sum().detach().cpu())
                log_tokens += token_count
                log_windows += window_count
                log_micro_steps += 1
                source_lines_seen.update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())
                accumulate_metrics(
                    metric_sums,
                    loss,
                    outputs,
                    parallel_loss,
                    batch,
                    config,
                    args.parallel_alignment_weight,
                )
                current_batch, current_batch_seen, wait_seconds = batch_source.next_after_step(
                    current_batch,
                    current_batch_seen,
                )
                data_seconds += wait_seconds
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step
            log_steps += 1

            should_log = step % args.log_every == 0 or step == start_step + 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_micro_steps, 1) for key, value in metric_sums.items()}
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
                        evaluate_parallel(
                            model,
                            eval_loader,
                            teacher,
                            device,
                            compile_mode,
                            autocast_enabled,
                            cuda_prefetch,
                            args.max_eval_batches,
                            args.parallel_alignment_weight,
                        )
                    )
                print(format_parallel_log(step, averaged), flush=True)
                last_metrics = averaged
                log_start = time.perf_counter()
                log_tokens = 0
                log_windows = 0
                log_steps = 0
                log_micro_steps = 0
                data_seconds = 0.0
                compute_seconds = 0.0
                metric_sums = empty_metric_sums()

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
        batch_source.close()
        save_interrupted()
        return
    except RuntimeError as error:
        batch_source.close()
        if not is_dataloader_worker_exit(error):
            raise
        save_interrupted()
        return

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
