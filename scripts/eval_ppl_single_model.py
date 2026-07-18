import argparse
import json
from pathlib import Path

from env_utils import load_project_dotenv
from any_precision.evaluate import eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity for a single packed model.")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--datasets", type=str, default="wikitext2,c4")
    parser.add_argument("--chunk_size", type=int, default=4096)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_project_dotenv(repo_root)

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    tokenizer_type, tokenizer, model = eval.auto_model_load(args.model_path)
    ppl_results = eval.evaluate_ppl(
        model,
        tokenizer,
        dataset_names,
        verbose=True,
        chunk_size=args.chunk_size,
        tokenizer_type=tokenizer_type,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "chunk_size": args.chunk_size,
        "datasets": dataset_names,
        "ppl": ppl_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
