import argparse
import random
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (  # noqa: E402
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cuda_sync,
    effective_compile_mode,
    validate_compile_environment,
)
from dil_data import load_hybrid_tokenizer, make_dil_batch_loader  # noqa: E402
from dilnaz_config import DIL_MODEL_DEFAULTS, DIL_TRAIN_DEFAULTS  # noqa: E402
from models.configuration_dil import DilConfig  # noqa: E402
from models.modeling_dil import Dil  # noqa: E402
from parallel_dil_data import (  # noqa: E402
    DEFAULT_PARALLEL_NLLB_MODEL,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    ParallelDilBatchDataset,
    ParallelNllbTeacher,
    parallel_total_loss,
)
from tokenization import default_vocab_path  # noqa: E402
from train_dil import (  # noqa: E402
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
    parser.add_argument("--semantic-normalizer-momentum", type=float, default=DIL_MODEL_DEFAULTS["semantic_normalizer_momentum"])
    parser.add_argument("--semantic-normalizer-eps", type=float, default=DIL_MODEL_DEFAULTS["semantic_normalizer_eps"])
    parser.add_argument("--semantic-normalizer-z-clip", type=float, default=DIL_MODEL_DEFAULTS["semantic_normalizer_z_clip"])
    parser.add_argument("--normalized-log-std-min", type=float, default=DIL_MODEL_DEFAULTS["normalized_log_std_min"])
    parser.add_argument("--normalized-log-std-max", type=float, default=DIL_MODEL_DEFAULTS["normalized_log_std_max"])
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
    if args.ce_weight <= 0:
        raise ValueError("--ce-weight must be > 0")
    if args.parallel_alignment_weight < 0:
        raise ValueError("--parallel-alignment-weight must be >= 0")
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
    if args.semantic_normalizer_momentum <= 0.0 or args.semantic_normalizer_momentum > 1.0:
        raise ValueError("--semantic-normalizer-momentum must be in (0, 1]")
    if args.semantic_normalizer_eps <= 0.0:
        raise ValueError("--semantic-normalizer-eps must be > 0")
    if args.semantic_normalizer_z_clip <= 0.0:
        raise ValueError("--semantic-normalizer-z-clip must be > 0")
    if args.normalized_log_std_min >= args.normalized_log_std_max:
        raise ValueError("--normalized-log-std-min must be smaller than --normalized-log-std-max")


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
        semantic_normalizer_momentum=args.semantic_normalizer_momentum,
        semantic_normalizer_eps=args.semantic_normalizer_eps,
        semantic_normalizer_z_clip=args.semantic_normalizer_z_clip,
        normalized_log_std_min=args.normalized_log_std_min,
        normalized_log_std_max=args.normalized_log_std_max,
        tokenizer_vocab_file=args.tokenizer_vocab.name,
        nllb_model_name=args.nllb_model_name,
        nllb_src_lang=args.source_lang,
    )


def teacher_dtype_for(device: torch.device, autocast_enabled: bool) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if autocast_enabled else torch.float16


def format_parallel_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"dil={metrics['dil_loss']:.4f}",
        f"parallel={metrics['parallel']:.4f}",
        f"parallel_w={metrics['parallel_weighted']:.4f}",
        f"ce={metrics['ce']:.4f}",
        f"ce_w={metrics['ce_weighted']:.4f}",
        f"kl={metrics['kl']:.2f}",
        f"kl_w={metrics['kl_weighted']:.4f}",
        f"geom_l1={metrics['geom_l1']:.4f}",
        f"geom_l2={metrics['geom_l2']:.4f}",
        f"geom_l3={metrics['geom_l3']:.4f}",
        f"geom_l4={metrics['geom_l4']:.4f}",
        f"geom_mean={metrics['geom_mean']:.4f}",
        f"var={metrics['var']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
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
        "ce": 0.0,
        "ce_weighted": 0.0,
        "kl": 0.0,
        "kl_weighted": 0.0,
        "geom_l1": 0.0,
        "geom_l2": 0.0,
        "geom_l3": 0.0,
        "geom_l4": 0.0,
        "geom_mean": 0.0,
        "var": 0.0,
        "byte_acc": 0.0,
        "align_groups": 0.0,
    }


def accumulate_metrics(metric_sums: dict[str, float], loss, outputs, parallel_loss, batch, config, weight: float):
    metric_sums["loss"] += float(loss.detach().cpu())
    metric_sums["dil_loss"] += float(outputs.loss.detach().cpu())
    metric_sums["parallel"] += float(parallel_loss.detach().cpu())
    metric_sums["parallel_weighted"] += float((parallel_loss * weight).detach().cpu())
    metric_sums["ce"] += float(outputs.ce_loss.detach().cpu())
    metric_sums["ce_weighted"] += float((outputs.ce_loss * config.ce_weight).detach().cpu())
    metric_sums["kl"] += float(outputs.kl_loss.detach().cpu())
    metric_sums["kl_weighted"] += float((outputs.kl_loss * config.kl_weight).detach().cpu())
    layer_losses = outputs.layer_geometry_losses.detach().cpu().tolist()
    for idx in range(4):
        metric_sums[f"geom_l{idx + 1}"] += float(layer_losses[idx]) if idx < len(layer_losses) else 0.0
    metric_sums["geom_mean"] += float(outputs.mean_geometry_loss.detach().cpu())
    metric_sums["var"] += float(outputs.variance_loss.detach().cpu())
    metric_sums["byte_acc"] += float(outputs.byte_acc.detach().cpu())
    metric_sums["align_groups"] += float(batch["parallel_alignment_scores"].shape[0])


@torch.no_grad()
def evaluate_parallel(
    model,
    eval_loader,
    teacher: ParallelNllbTeacher,
    device,
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
        with autocast_context(autocast_enabled):
            outputs = model(**model_inputs(batch))
            loss, parallel_loss = parallel_total_loss(outputs, batch, parallel_alignment_weight)
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
            batch = current_batch
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_enabled):
                outputs = model(**model_inputs(batch))
                loss, parallel_loss = parallel_total_loss(outputs, batch, args.parallel_alignment_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step
            current_batch, current_batch_seen, wait_seconds = batch_source.next_after_step(
                current_batch,
                current_batch_seen,
            )
            data_seconds += wait_seconds

            log_tokens += int(batch["labels"].ne(-100).sum().detach().cpu())
            log_windows += int(batch["labels"].shape[0])
            log_steps += 1
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
                        evaluate_parallel(
                            model,
                            eval_loader,
                            teacher,
                            device,
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
