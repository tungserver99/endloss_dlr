def _require_hf_datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for raw Hugging Face dataset loading. "
            "Install it with `pip install datasets` if you want raw calibration/eval datasets."
        ) from exc
    return get_dataset_config_names, load_dataset

import gzip
import io
import json
import random
import numpy as np
import logging
import torch
from tqdm import tqdm
import os
import shutil
import subprocess
from urllib.request import urlopen

DEFAULT_REDPAJAMA_REPO = "togethercomputer/RedPajama-Data-1T"
DEFAULT_REDPAJAMA_SOURCE = "cache"
TOGETHER_REDPAJAMA_URLS = "https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt"
GUIDEDQUANT_RELEASE_BASE = "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0"


def _guidedquant_calibration_filename(model_name: str, dataset_name: str, seq_len: int, num_samples: int) -> str | None:
    if dataset_name == "redpajama" and seq_len == 4096 and num_samples == 1024:
        if model_name in {"Llama-2-7b", "Llama-2-7b-hf", "Llama-2-13b", "Llama-2-13b-hf", "Llama-2-70b", "Llama-2-70b-hf"}:
            return "Llama-2-7b-hf-redpajama_s1024_blk4096.pt"
        if model_name in {"Meta-Llama-3-8B", "Meta-Llama-3-70B"}:
            return "Meta-Llama-3-8B-redpajama_s1024_blk4096.pt"

    if dataset_name == "wikitext2" and seq_len == 2048 and num_samples == 128:
        if model_name in {"Llama-2-7b", "Llama-2-7b-hf", "Llama-2-13b", "Llama-2-13b-hf", "Llama-2-70b", "Llama-2-70b-hf"}:
            return "Llama-2-7b-hf-wikitext2_s128_blk2048.pt"

    return None

def _get_wikitext2(split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    _, load_dataset = _require_hf_datasets()
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
    return data['text']


def _get_ptb(split, slice_unk=True):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for ptb"

    _, load_dataset = _require_hf_datasets()
    data = load_dataset('ptb_text_only', 'penn_treebank', split=split,
                        trust_remote_code=True)
    data_list = data['sentence']

    if slice_unk:
        data_list = [s.replace('<unk>', '< u n k >') for s in data_list]

    return data_list


def _get_c4(split):
    assert split in ['train', 'validation'], f"Unknown split {split} for c4"

    _, load_dataset = _require_hf_datasets()

    if split == 'train':
        _, load_dataset = _require_hf_datasets()
        data = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
            trust_remote_code=True
        )
    else:
        assert split == 'validation'
        _, load_dataset = _require_hf_datasets()
        data = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
            trust_remote_code=True
        )

    return data['text']


def _get_pileval(split):
    if split != 'validation':
        logging.warning(f"Pileval only has a validation split, but got split={split}. Using validation split.")
    _, load_dataset = _require_hf_datasets()
    data = load_dataset("mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True)

    return data['text']


def _get_redpajama(split):
    assert split in ['train'], "RedPajama only has a train split"
    get_dataset_config_names, load_dataset = _require_hf_datasets()
    dataset_repo = os.environ.get("REDPAJAMA_DATASET_REPO", DEFAULT_REDPAJAMA_REPO)
    logging.info(f"Loading RedPajama from {dataset_repo}")
    load_errors = []

    def _try_load(config_name=None):
        try:
            kwargs = {"path": dataset_repo, "split": split}
            if config_name is not None:
                kwargs["name"] = config_name
            return load_dataset(**kwargs)
        except Exception as exc:
            load_errors.append((config_name, exc))
            return None

    data = _try_load()
    if data is not None:
        return data['text']

    try:
        config_names = get_dataset_config_names(dataset_repo)
    except Exception as exc:
        load_errors.append(("__configs__", exc))
        config_names = []

    preferred_configs = []
    for preferred in ("plain_text", "default"):
        if preferred in config_names:
            preferred_configs.append(preferred)
    preferred_configs.extend([name for name in config_names if name not in preferred_configs])

    for config_name in preferred_configs:
        data = _try_load(config_name)
        if data is not None:
            logging.info(f"Loaded RedPajama config '{config_name}' from {dataset_repo}")
            return data['text']

    error_lines = [
        f"config={config_name!r}: {type(exc).__name__}: {exc}"
        for config_name, exc in load_errors
    ]
    raise RuntimeError(
        "Unable to load RedPajama via datasets. Tried direct load and available configs.\n"
        + "\n".join(error_lines)
    )


def _should_use_together_stream_fallback(dataset_repo, exc):
    if not isinstance(dataset_repo, str) or "togethercomputer/RedPajama-Data-1T" not in dataset_repo:
        return False
    message = str(exc)
    return (
        "Dataset scripts are no longer supported" in message
        or "RedPajama-Data-1T.py" in message
    )


def _fetch_together_redpajama_urls():
    logging.info(f"Fetching Together RedPajama shard list from {TOGETHER_REDPAJAMA_URLS}")
    with urlopen(TOGETHER_REDPAJAMA_URLS, timeout=60) as response:
        payload = response.read().decode("utf-8")
    urls = [line.strip() for line in payload.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError("Together RedPajama urls.txt was empty")
    return urls


def _iter_together_redpajama_texts(urls):
    for url in urls:
        logging.info(f"Streaming Together RedPajama shard: {url}")
        with urlopen(url, timeout=120) as response:
            raw_stream = response
            if url.endswith(".gz"):
                text_stream = io.TextIOWrapper(gzip.GzipFile(fileobj=raw_stream), encoding="utf-8")
            else:
                text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8")

            with text_stream:
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text")
                    if isinstance(text, str) and text:
                        yield text


def _build_redpajama_tokens_from_together(tokenizer, seq_len, num_samples, seed=None):
    rng = random.Random(seed)
    urls = _fetch_together_redpajama_urls()
    rng.shuffle(urls)

    samples = []
    pbar = tqdm(total=num_samples, desc="Streaming Together RedPajama")
    for text in _iter_together_redpajama_texts(urls):
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:
            continue

        seq_start = rng.randint(0, len(tokens) - seq_len)
        samples.append(tokens[seq_start:seq_start + seq_len])
        pbar.update(1)

        if len(samples) >= num_samples:
            break

    pbar.close()

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Together RedPajama fallback produced only {len(samples)} / {num_samples} samples"
        )

    return samples


def _download_redpajama_cache(save_path, model_name, seq_len, num_samples, dataset_name="redpajama"):
    if os.path.isfile(save_path):
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    filename = _guidedquant_calibration_filename(model_name, dataset_name, seq_len, num_samples)
    if filename is None:
        raise FileNotFoundError(
            "No built-in GuidedQuant calibration cache artifact is available for "
            f"model={model_name}, dataset={dataset_name}, num_samples={num_samples}, seq_len={seq_len}. "
            "Use redpajama_source='raw' or provide the cache file manually."
        )
    url = f"{GUIDEDQUANT_RELEASE_BASE}/{filename}"

    downloader = shutil.which("wget")
    if downloader is not None:
        cmd = [downloader, "-O", save_path, url]
    else:
        downloader = shutil.which("curl")
        if downloader is None:
            raise RuntimeError(
                "Need either 'wget' or 'curl' to auto-download RedPajama cache."
            )
        cmd = [downloader, "-L", url, "-o", save_path]

    logging.info(f"Downloading RedPajama cache to {save_path}")
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        tokens = tokens[:seq_len]

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples

def _sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()
    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        seq_start = random.randint(0, len(tokens) - seq_len)

        tokens = tokens[seq_start:seq_start + seq_len]
        assert tokens.shape[-1] == seq_len, f"Token length {len(tokens)} != seq_len {seq_len}"

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()
    return samples


def _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
    f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        idx = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        
        if selected_indices:
            closest_idx = min(selected_indices, key=lambda x: abs(x - idx), default=idx)
            if idx <= closest_idx + seq_len and idx >= closest_idx - seq_len:
                continue

        j = idx + seq_len
        inp = trainenc.input_ids[:, idx:j]
        tokens = inp.clone()
        tokens = tokens.squeeze(0)

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def _get_dataset(dataset_name, split):
    if dataset_name == 'wikitext2':
        return _get_wikitext2(split)
    elif dataset_name == 'ptb':
        return _get_ptb(split)
    elif dataset_name == 'c4':
        return _get_c4(split)
    elif dataset_name == 'pileval':
        return _get_pileval(split)
    elif dataset_name == 'redpajama':
        return _get_redpajama(split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(
    dataset_name,
    split,
    tokenizer,
    seq_len,
    num_samples,
    save_path=None,
    seed=None,
    redpajama_source=DEFAULT_REDPAJAMA_SOURCE,
):

    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Loading tokens from {save_path}")
        return torch.load(save_path)

    if dataset_name == 'redpajama' and redpajama_source == 'cache':
        if save_path is None:
            raise ValueError("save_path is required when redpajama_source='cache'")
        model_name = tokenizer.name_or_path.rstrip("/").split("/")[-1]
        _download_redpajama_cache(save_path, model_name, seq_len, num_samples, dataset_name=dataset_name)
        logging.info(f"Loading downloaded RedPajama cache from {save_path}")
        return torch.load(save_path)

    logging.info(f"Fetching dataset: {dataset_name}")
    try:
        texts = _get_dataset(dataset_name, split)
    except RuntimeError as exc:
        dataset_repo = os.environ.get("REDPAJAMA_DATASET_REPO", DEFAULT_REDPAJAMA_REPO)
        if dataset_name == 'redpajama' and _should_use_together_stream_fallback(dataset_repo, exc):
            logging.warning(
                "Falling back to direct Together RedPajama shard streaming because "
                "the installed datasets version no longer supports this dataset script."
            )
            tokens = _build_redpajama_tokens_from_together(
                tokenizer=tokenizer,
                seq_len=seq_len,
                num_samples=num_samples,
                seed=seed,
            )
            if save_path is not None:
                logging.info(f"Saving tokens to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(tokens, save_path)
            return tokens
        raise

    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    if dataset_name == 'wikitext2':
        tokens = _sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)
    elif dataset_name == 'redpajama':
        # Following PV-Tuning Github.
        tokens = _sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed)
    else:
        tokens = _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)

    if save_path is not None:
        logging.info(f"Saving tokens to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tokens, save_path)

    return tokens
