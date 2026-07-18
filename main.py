import argparse
from pathlib import Path

from env_utils import load_project_dotenv
from any_precision.quantization.end_loss_dlr import hybrid_end_loss_quantize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full End-Loss DLR scalar quantization pipeline"
    )
    parser.add_argument("model", type=str, help="HF model repo or local model path")
    parser.add_argument("--bits", type=int, default=3, help="Target bit-width")
    parser.add_argument("--yaml_path", type=str, help="Architecture config yaml file")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--dataset", type=str, default="redpajama", help="Calibration dataset")
    parser.add_argument("--seq_len", type=int, default=4096, help="Calibration sequence length")
    parser.add_argument("--num_examples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--redpajama_source", type=str, default="cache", choices=["cache", "raw"])
    parser.add_argument("--overwrite_quantize", action="store_true")
    parser.add_argument("--overwrite_pack", action="store_true")
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--calibration_batch_size", type=int, default=1)
    parser.add_argument("--fisher_probes", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num_output_groups", type=int, default=8)
    parser.add_argument("--row_batch_size", type=int, default=128)
    parser.add_argument("--max_outer_iters", type=int, default=8)
    parser.add_argument("--rel_tol", type=float, default=1e-7)
    parser.add_argument("--lambda_safety", type=float, default=1.01)
    parser.add_argument("--tie_tol", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    load_project_dotenv(Path(__file__).resolve().parent)
    args = build_parser().parse_args()

    hybrid_end_loss_quantize(
        model=args.model,
        bits=args.bits,
        yaml_path=args.yaml_path,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        redpajama_source=args.redpajama_source,
        overwrite_quantize=args.overwrite_quantize,
        overwrite_pack=args.overwrite_pack,
        random_state=args.random_state,
        calibration_batch_size=args.calibration_batch_size,
        fisher_probes=args.fisher_probes,
        beta=args.beta,
        rank=args.rank,
        num_output_groups=args.num_output_groups,
        row_batch_size=args.row_batch_size,
        max_outer_iters=args.max_outer_iters,
        rel_tol=args.rel_tol,
        lambda_safety=args.lambda_safety,
        tie_tol=args.tie_tol,
    )
