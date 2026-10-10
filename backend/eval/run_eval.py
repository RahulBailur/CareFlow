"""Evaluate CareBot against the labelled utterances.

    python eval/run_eval.py --classifier                      # keywords + embeddings if installed
    python eval/run_eval.py --classifier --no-embeddings      # what CI runs: no model download
    python eval/run_eval.py --classifier --set heldout
    python eval/run_eval.py --classifier --min-accuracy 0.85  # exit 1 below the gate

Two numbers matter. "Accuracy" is how often the classifier's best guess is right.
"Settled locally" is how many turns it was confident enough to answer without the LLM,
and how often it was right on those: a confident wrong answer is the costly kind.

`utterances.jsonl` was written alongside the keyword rules, so it flatters them.
`utterances_heldout.jsonl` was written before the rules were first run.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from agents.embeddings import load_embedder
from agents.intent_classifier import IntentClassifier
from config import get_settings

EVAL_DIR = Path(__file__).parent
SETS = {"dev": "utterances.jsonl", "heldout": "utterances_heldout.jsonl"}


def load(name: str) -> list[dict[str, str]]:
    lines = (EVAL_DIR / SETS[name]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(classifier: IntentClassifier, rows: list[dict[str, str]], verbose: bool) -> float:
    total: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    local = local_correct = 0
    for row in rows:
        result = classifier.classify(row["text"])
        right = result.intent == row["intent"]
        for key in ("all", row["lang"]):
            total[key] += 1
            correct[key] += right
        if not result.needs_llm:
            local += 1
            local_correct += right
        if verbose and (not right or result.needs_llm):
            flag = "WRONG" if not right else "to LLM"
            print(
                f"  {flag:6} {row['id']:9} want {row['intent']:8} got {result.intent:8} "
                f"({result.source}, {result.confidence:.2f})"
            )

    for key in ("all", *sorted(k for k in total if k != "all")):
        print(f"  {key:6} accuracy {correct[key]}/{total[key]} = {correct[key] / total[key]:.1%}")
    print(f"  settled locally {local}/{total['all']} = {local / total['all']:.1%}", end="")
    print(f", of which correct {local_correct}/{local}" if local else "")
    return correct["all"] / total["all"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CareBot on the labelled utterances")
    parser.add_argument("--classifier", action="store_true", help="evaluate the intent classifier")
    parser.add_argument("--set", choices=[*SETS, "all"], default="dev")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--verbose", "-v", action="store_true", help="list misses and hand-offs")
    args = parser.parse_args()
    if not args.classifier:
        parser.error("only --classifier is implemented so far")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    embedder = None if args.no_embeddings else load_embedder(get_settings().intent_embedding_model)
    classifier = IntentClassifier(embedder)
    print(f"Intent classifier: keywords {'+ embeddings' if embedder else 'only'}")

    gate_ok = True
    for name in SETS if args.set == "all" else [args.set]:
        print(f"{name} set ({SETS[name]})")
        accuracy = evaluate(classifier, load(name), args.verbose)
        if name == "dev" and accuracy < args.min_accuracy:
            print(f"FAIL: dev accuracy {accuracy:.1%} is below the gate {args.min_accuracy:.0%}")
            gate_ok = False
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
