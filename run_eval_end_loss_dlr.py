import argparse
import gc
import json
import os

import torch

from any_precision.evaluate import eval


parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default="results_endloss_dlr.json")
parser.add_argument("--cache_dir", type=str, default="./cache")
parser.add_argument("--downstream", action="store_true")
args = parser.parse_args()

datasets = ["wikitext2", "c4"]
tasks = ["boolq", "piqa", "social_iqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa"] if args.downstream else []
model_root = os.path.join(args.cache_dir, "endloss_dlr_quantized")
model_paths = [
    os.path.join(model_root, name)
    for name in sorted(os.listdir(model_root))
    if os.path.isdir(os.path.join(model_root, name))
]

if os.path.exists(args.output_file):
    with open(args.output_file, "r", encoding="utf-8") as handle:
        all_results = json.load(handle)
else:
    all_results = {}


def save_results(results_dict):
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


for model_path in model_paths:
    model_name = os.path.basename(model_path)
    tokenizer_type, tokenizer, model = eval.auto_model_load(model_path)
    ppl_results = eval.evaluate_ppl(model, tokenizer, datasets, verbose=True, chunk_size=4096, tokenizer_type=tokenizer_type)
    all_results.setdefault(model_name, {})["ppl"] = ppl_results
    save_results(all_results)
    if tasks:
        task_results = eval.run_lm_eval(tokenizer, model, tasks)
        all_results.setdefault(model_name, {})["lm-eval"] = task_results
        save_results(all_results)
    del model
    del tokenizer
    cleanup()
