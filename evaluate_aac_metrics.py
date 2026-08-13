import os
import json
import argparse
from aac_metrics import evaluate # type: ignore

def evaluate_aac_metrics(candidates, references, output_path=None):
    """MUST RUN on the versions specified in env_aac_metrics.yml 

    Args:
        candidates (candidates): List of Candidate captions (strings) to evaluate.
        references (_type_): List of Reference captions (strings or list of strings) corresponding to each candidate.
        output_path (_type_, optional): Where to save the evaluation results as a JSON file. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # 1. Clean candidates and ensure string type
    cleaned_candidates: list[str] = [
        str(c).replace("clotho > caption: ", "").strip() for c in candidates
    ]

    # 2. Process references safely without over-nesting
    mult_ref: list[list[str]] = []
    for ref in references:
        if isinstance(ref, list):
            # Already a list of reference strings for one audio file
            mult_ref.append([str(r).replace("clotho > caption: ", "").strip() for r in ref])
        else:
            # Single reference string wrapped into a list
            mult_ref.append([str(ref).replace("clotho > caption: ", "").strip()])

    # 3. Validate alignment
    if len(cleaned_candidates) != len(mult_ref):
        raise ValueError(
            f"Mismatch in length: {len(cleaned_candidates)} candidates vs {len(mult_ref)} reference sets."
        )

    eval_kwargs = {
        "metrics": "dcase2024"
    }

    java_path: str = os.path.expanduser(
        "~/.cache/x64-jdk11/amazon-corretto-11.jdk/Contents/Home/bin/java"
    )
    print("\n--- Evaluating caption generation ---")
    print(f"Evaluating {len(cleaned_candidates)} candidates against {len(mult_ref)} reference sets")

    # 4. Run evaluation with CLEANED candidates
    corpus_scores, _ = evaluate(
        cleaned_candidates,
        mult_ref,
        **eval_kwargs,
        java_path=java_path
    )

    # Convert tensor/numpy floats to standard Python floats for JSON serialization
    formatted_scores = {metric.upper(): float(score) for metric, score in corpus_scores.items()}

    for metric, score in formatted_scores.items():
        print(f"{metric}: {score:.4f}")

    # 5. Save results downstream if output path is specified
    if output_path:
        with open(output_path, "w") as f:
            json.dump(formatted_scores, f, indent=4)
        print(f"\nSaved metric scores to {output_path}")

    return formatted_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="candidates.json", help="Path to json file with candidates and references")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save calculated scores as JSON")
    args = parser.parse_args()

    print(f"Loading candidates and references from {args.json_file}...")
    with open(args.json_file, "r") as json_file:
        data = json.load(json_file)
        candidates = data["candidates"]
        references = data["references"]

    # Determine default output file name if not explicitly passed
    output_file = args.output_file
    if output_file is None:
        output_file = args.json_file.replace(".json", "_scores.json")

    evaluate_aac_metrics(candidates, references, output_path=output_file)
    exit(0)