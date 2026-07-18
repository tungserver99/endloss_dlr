import argparse
from pathlib import Path

from env_utils import load_project_dotenv
from any_precision.quantization.endloss_main import any_precision_quantize


if __name__ == "__main__":
    load_project_dotenv(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("model", type=str, help="The model to quantize")
    parser.add_argument("--seed_precision", type=int, help="The precision to quantize the seed to")
    parser.add_argument("--parent_precision", type=int, help="The precision to quantize the parent to")
    parser.add_argument("--mode", type=str, default="pack", help="The mode to run in")
    parser.add_argument("--yaml_path", type=str, help="The path to the architecture config yaml file")
    parser.add_argument("--cache_dir", type=str, help="The directory to cache results in")
    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--seq_len", type=int, help="The sequence length to use")
    parser.add_argument("--num_examples", type=int, help="The number of examples to use")
    parser.add_argument("--redpajama_source", type=str, choices=["cache", "raw"],
                        help="How to source RedPajama calibration tokens")
    parser.add_argument("--redpajama_dataset_repo", type=str,
                        help="HF dataset repo to use when redpajama_source=raw")
    parser.add_argument("--cpu_count", type=int, help="The number of CPUs to use for parallelization")
    parser.add_argument("--overwrite_tokens", action="store_true",
                        help="Whether to overwrite the tokens stored to disk")
    parser.add_argument('--overwrite_gradients', action="store_true",
                        help="Whether to overwrite the gradients stored to disk")
    parser.add_argument("--overwrite_quantize", action="store_true",
                        help="Whether to overwrite the parent model stored to disk")
    parser.add_argument("--overwrite_pack", action="store_true",
                        help="Whether to overwrite the packed model stored to disk")
    parser.add_argument("--random_state", type=int,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    parser.add_argument("--dns", action="store_true",
                        help="REALLY Experimental: Whether to run Dense & Sparse quantization")
    parser.add_argument("--num_groups", type=int, default=None,
                        help="Number of groups $g$ to use for GuidedQuant Hessian")
    parser.add_argument("--sub_saliency", nargs='+', type=int, default=None,
                        help="(start, end) of layers to use for saliency saving")
    parser.add_argument("--skip_save_gradients", action="store_true",
                        help="Whether to skip saving gradients")
    parser.add_argument("--beta", type=float, help="Hybrid loss mixing coefficient")
    parser.add_argument("--rank", type=int, help="Low-rank curvature rank")
    parser.add_argument("--num_output_groups", type=int, help="Number of output groups for shared curvature")
    parser.add_argument("--calibration_batch_size", type=int, help="Calibration batch size on GPU")
    parser.add_argument("--fisher_probes", type=int, help="Number of Fisher probe batches")
    parser.add_argument("--gradient_num_examples", type=int,
                        help="Number of calibration examples to use for NLL-gradient collection")
    parser.add_argument("--stats_layer_chunk_size", type=int,
                        help="Number of transformer layers to keep trainable at once during statistics collection")
    parser.add_argument("--max_outer_iters", type=int, help="Maximum alternating iterations for scalar DLR quantization")
    parser.add_argument("--rel_tol", type=float, help="Relative loss-drop tolerance for stopping")
    parser.add_argument("--lambda_safety", type=float, help="Safety multiplier for the MM spectral majorizer")
    parser.add_argument("--tie_tol", type=float, help="Tolerance for current-label tie breaking in MM assignment")

    args = parser.parse_args()
    args.sub_saliency = tuple(args.sub_saliency) if args.sub_saliency else None
    any_precision_quantize(**{k: v for k, v in args.__dict__.items() if v is not None})
