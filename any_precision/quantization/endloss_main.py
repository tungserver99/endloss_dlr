import datetime
import logging
import os
import shutil
import sys

import torch

from ..analyzer import get_analyzer
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET,
    DEFAULT_NUM_EXAMPLES,
    DEFAULT_PARENT_PRECISION,
    DEFAULT_SEQ_LEN,
    DEFAULT_SEED_PRECISION,
)
from .datautils import get_tokens
from .end_loss_dlr.config import EndLossDLRConfig
from .end_loss_dlr.layer_quantizer_cuda import collect_end_loss_statistics, quantize_model
from .end_loss_dlr.serialization import save_layer_artifacts, save_metadata
from .pack import pack


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _format_optional_int(value):
    return "all" if value is None else str(value)


def _nll_cache_suffix(calibration_batch_size: int, gradient_num_examples, stats_layer_chunk_size: int) -> str:
    return (
        f"cb{calibration_batch_size}_"
        f"gn{_format_optional_int(gradient_num_examples)}_"
        f"lc{stats_layer_chunk_size}_v2"
    )


def _stats_cache_suffix(rank: int, num_output_groups: int, fisher_probes: int, calibration_batch_size: int, gradient_num_examples, stats_layer_chunk_size: int) -> str:
    return (
        f"r{rank}_og{num_output_groups}_fp{fisher_probes}_"
        f"{_nll_cache_suffix(calibration_batch_size, gradient_num_examples, stats_layer_chunk_size)}_v4"
    )


def _setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/quantize_log_{timestamp}.txt"),
        ]
    )


def _normalize_tokens(tokens):
    if isinstance(tokens, list):
        return torch.stack([item.long().cpu() for item in tokens], dim=0)
    if isinstance(tokens, torch.Tensor) and tokens.ndim == 3 and tokens.shape[1] == 1:
        return tokens[:, 0, :]
    if isinstance(tokens, torch.Tensor):
        return tokens.long().cpu()
    raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")


def any_precision_quantize(
        model,
        seed_precision=DEFAULT_SEED_PRECISION,
        parent_precision=DEFAULT_PARENT_PRECISION,
        mode='pack',
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
        dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
        redpajama_source="cache",
        redpajama_dataset_repo=None,
        cpu_count=None,
        overwrite_tokens=False,
        overwrite_gradients=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        random_state=None,
        dns=False,
        num_groups=None,
        sub_saliency=None,
        skip_save_gradients=False,
        beta=0.5,
        rank=4,
        num_output_groups=8,
        calibration_batch_size=1,
        fisher_probes=16,
        gradient_num_examples=None,
        stats_layer_chunk_size=8,
        max_outer_iters=8,
        rel_tol=1e-7,
        lambda_safety=1.01,
        tie_tol=0.0,
):
    del cpu_count, dns, num_groups, sub_saliency, skip_save_gradients

    _setup_logging()

    assert mode in ['tokens', 'gradients', 'quantize', 'pack'], \
        "mode must be one of 'tokens', 'gradients', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."

    if seed_precision != parent_precision:
        logging.warning(
            "End-Loss DLR scalar quantization does not use seed/upscale hierarchy. "
            f"Using target bit-width parent_precision={parent_precision} and ignoring seed_precision={seed_precision}."
        )

    if overwrite_tokens and not overwrite_gradients:
        logging.warning("Statistics need to be recalculated if tokens are recalculated. Setting overwrite_gradients to True.")
        overwrite_gradients = True
    if overwrite_gradients and not overwrite_quantize:
        logging.warning("Quantized cache needs to be recalculated if statistics are recalculated. Setting overwrite_quantize to True.")
        overwrite_quantize = True
    if overwrite_quantize and not overwrite_pack:
        logging.warning("Packed model needs to be recalculated if quantized cache is recalculated. Setting overwrite_pack to True.")
        overwrite_pack = True

    if mode == 'tokens':
        logging.info("Running: [Tokens]")
    elif mode == 'gradients':
        logging.info("Running: [Tokens -> End-loss Statistics]")
    elif mode == 'quantize':
        logging.info("Running: [Tokens -> End-loss Statistics -> Quantize]")
    else:
        logging.info("Running: [Tokens -> End-loss Statistics -> Quantize -> Pack]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    logging.info(
        f"Running End-Loss DLR Scalar Quantization on {model_name} with target precision {parent_precision} "
        f"using {dataset} for end-loss statistics"
    )

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)

    tokens_cache_path = f"{cache_dir}/tokens/{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt"
    nll_cache_suffix = _nll_cache_suffix(calibration_batch_size, gradient_num_examples, stats_layer_chunk_size)
    stats_cache_suffix = _stats_cache_suffix(rank, num_output_groups, fisher_probes, calibration_batch_size, gradient_num_examples, stats_layer_chunk_size)
    gradients_cache_path = f"{cache_dir}/gradients/{model_name}-{dataset}_s{num_examples}_blk{seq_len}-{stats_cache_suffix}.pt"
    nll_gradients_cache_path = f"{cache_dir}/gradients/{model_name}-{dataset}_s{num_examples}_blk{seq_len}-{nll_cache_suffix}.nll_tmp.pt"
    legacy_gradients_cache_path = f"{cache_dir}/gradients/{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt"
    legacy_nll_gradients_cache_path = f"{cache_dir}/gradients/{model_name}-{dataset}_s{num_examples}_blk{seq_len}.nll_tmp.pt"
    quantized_cache_path = (
        f"{cache_dir}/quantized/{model_name}-w{parent_precision}_orig{parent_precision}"
        f"-{dataset}_s{num_examples}_blk{seq_len}"
    )
    model_output_path = (
        f"{cache_dir}/packed/anyprec-{model_name}-w{parent_precision}_orig{parent_precision}"
        f"-{dataset}_s{num_examples}_blk{seq_len}"
    )

    logging.info(f"Tokens cache path: {tokens_cache_path}")
    logging.info(f"Statistics cache path: {gradients_cache_path}")
    logging.info(f"Temporary NLL cache path: {nll_gradients_cache_path}")
    logging.info(f"Legacy statistics cache path: {legacy_gradients_cache_path}")
    logging.info(f"Legacy temporary NLL cache path: {legacy_nll_gradients_cache_path}")
    logging.info(f"Quantized cache path: {quantized_cache_path}")
    logging.info(f"Model output path: {model_output_path}")

    logging.info("------------------- Get tokens -------------------")
    logging.info(f"Getting tokens for {dataset} with sequence length {seq_len} and {num_examples} examples")
    if dataset == "redpajama" and redpajama_dataset_repo is not None:
        os.environ["REDPAJAMA_DATASET_REPO"] = redpajama_dataset_repo

    if overwrite_tokens and os.path.exists(tokens_cache_path):
        logging.info(f"Detected cached tokens at {tokens_cache_path}. Will delete and recalculate.")
        os.remove(tokens_cache_path)

    tokens = get_tokens(
        dataset,
        "train",
        analyzer.tokenizer,
        seq_len,
        num_examples,
        tokens_cache_path,
        random_state,
        redpajama_source=redpajama_source,
    )
    tokens = _normalize_tokens(tokens)
    logging.info("Tokens loading complete.")

    if mode == 'tokens':
        return

    logging.info("------------------- End-loss Statistics -------------------")
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        logging.info(f"Detected cached statistics at {gradients_cache_path}. Will delete and recalculate.")
        os.remove(gradients_cache_path)
    if overwrite_gradients and os.path.exists(nll_gradients_cache_path):
        logging.info(f"Detected temporary NLL cache at {nll_gradients_cache_path}. Will delete and recalculate.")
        os.remove(nll_gradients_cache_path)

    if not os.path.exists(nll_gradients_cache_path) and os.path.exists(legacy_nll_gradients_cache_path):
        logging.info(
            "Reusing legacy temporary NLL cache by promoting %s -> %s",
            legacy_nll_gradients_cache_path,
            nll_gradients_cache_path,
        )
        os.makedirs(os.path.dirname(nll_gradients_cache_path), exist_ok=True)
        shutil.copy2(legacy_nll_gradients_cache_path, nll_gradients_cache_path)

    dlr_config = EndLossDLRConfig(
        bits=parent_precision,
        beta=beta,
        rank=rank,
        num_output_groups=num_output_groups,
        calibration_batch_size=calibration_batch_size,
        fisher_probes=fisher_probes,
        gradient_num_examples=gradient_num_examples,
        stats_layer_chunk_size=stats_layer_chunk_size,
        max_outer_iters=max_outer_iters,
        rel_tol=rel_tol,
        lambda_safety=lambda_safety,
        tie_tol=tie_tol,
        cache_dir=cache_dir,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=num_examples,
    )

    if os.path.exists(gradients_cache_path):
        logging.info(f"Loading cached end-loss statistics from {gradients_cache_path}")
        model_stats = torch.load(gradients_cache_path, map_location="cpu", weights_only=False).to(dlr_config.device)
    else:
        logging.info("Beginning end-loss statistics collection...")
        model_stats = collect_end_loss_statistics(
            model=analyzer.model,
            calibration_loader=tokens,
            config=dlr_config,
            analyzer=analyzer,
            nll_cache_path=nll_gradients_cache_path,
        )
        torch.save(model_stats, gradients_cache_path)
    logging.info("End-loss statistics complete.")

    if mode == 'gradients':
        logging.info("Keeping temporary NLL cache for future resume: %s", nll_gradients_cache_path)
        return

    logging.info("------------------- Quantize: End-Loss DLR Scalar -------------------")
    logging.info(f"Beginning {parent_precision}-bit End-Loss DLR Scalar Quantization...")
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        logging.info(f"Detected cached quantized folder at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)
    os.makedirs(quantized_cache_path, exist_ok=True)

    saved_layers, metadata = quantize_model(
        model=analyzer.model,
        stats=model_stats,
        config=dlr_config,
        analyzer=analyzer,
    )
    metadata.update(
        {
            "beta": beta,
            "rank": rank,
            "num_output_groups": num_output_groups,
            "max_outer_iters": max_outer_iters,
            "rel_tol": rel_tol,
            "lambda_safety": lambda_safety,
            "tie_tol": tie_tol,
            "quantized_cache": quantized_cache_path,
            "packed_output": model_output_path,
        }
    )
    for layer_idx, layer_codebooks, layer_labels in saved_layers:
        save_layer_artifacts(quantized_cache_path, layer_idx, layer_codebooks, layer_labels)
    save_metadata(quantized_cache_path, metadata)

    if mode == 'quantize':
        if os.path.exists(nll_gradients_cache_path):
            os.remove(nll_gradients_cache_path)
            logging.info("Removed temporary NLL cache after successful quantization: %s", nll_gradients_cache_path)
        return

    analyzer.drop_original_weights()
    logging.info("Quantization complete.")

    logging.info("------------------- Pack -------------------")
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and re-pack.")
            shutil.rmtree(model_output_path)
        else:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return

    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=parent_precision,
        parent_precision=parent_precision,
        cpu_count=1,
        dns=False,
    )
    logging.info("Packing complete.")
    if os.path.exists(nll_gradients_cache_path):
        os.remove(nll_gradients_cache_path)
        logging.info("Removed temporary NLL cache after successful pipeline: %s", nll_gradients_cache_path)

