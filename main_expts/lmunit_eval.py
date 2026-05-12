#!/usr/bin/env python3
"""
LMUnit Evaluation Script

Evaluates responses using LMUnit with natural language unit tests.
Uses the official lmunit library as shown in:
https://huggingface.co/ContextualAI/LMUnit-qwen2.5-72b

Usage:
    python lmunit_eval.py --file <path> --model ContextualAI/LMUnit-qwen2.5-72b --tp_size 4
"""

import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from lmunit import LMUnit
from vllm import SamplingParams


UNIT_TESTS = [
    "Does the response directly and effectively address the user's request?",
    "Is the information in the response correct and reliable?",
    "Is the response well-structured, clear, and fluent?",
    "Does the response appropriately address the full scope of the question?",
    "Is the response free from harmful, biased, or inappropriate content?",
]


@dataclass
class LMUnitScore:
    """Score result for a single unit test."""
    unit_test: str
    score: float
    token_probs: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a response."""
    idx: int
    prompt: str
    response: str
    scores: list  # List of LMUnitScore
    avg_score: float


def load_results(filepath: str) -> dict:
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_lmunit_prompt(query: str, response: str, unit_test: str) -> str:
    """
    Format prompt in LMUnit's expected format.
    
    From: https://huggingface.co/ContextualAI/LMUnit-qwen2.5-72b
    """
    return f"Query: {query}\n\nResponse: {response}\n\nUnit Test: {unit_test}"


class LMUnitEvaluator:
    """LMUnit evaluator using the official lmunit library."""
    
    def __init__(
        self,
        model_path: str = "ContextualAI/LMUnit-qwen2.5-72b",
        tp_size: int = 4,
        debug: bool = False,
    ):
        """
        Initialize LMUnit evaluator.
        
        Args:
            model_path: HuggingFace model path
            tp_size: Tensor parallelism size
            debug: If True, print debug information
        """
        self.model_path = model_path
        self.debug = debug
        
        print(f"Loading LMUnit model: {model_path}")
        print(f"Tensor parallelism: {tp_size}")
        
        # Initialize using the official lmunit library
        self.model = LMUnit(
            model_path=model_path,
            tp_size=tp_size,
            gpu_memory_utilization=0.8,  # Adjust as needed
        )
        
        # Sampling params as per the HuggingFace example
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            logprobs=20,
        )
        
        print("Model loaded successfully!")
    
    def score(self, query: str, response: str, unit_test: str) -> LMUnitScore:
        """
        Score a response against a unit test.

        Args:
            query: The original user query/prompt
            response: The model's response to evaluate
            unit_test: The evaluation criterion

        Returns:
            LMUnitScore with score on 1-5 scale
        """
        content = format_lmunit_prompt(query, response, unit_test)

        # Apply chat template for Qwen model
        messages = [{"role": "user", "content": content}]
        prompt = self.model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if self.debug:
            print(f"    [DEBUG] Formatted prompt:\n{prompt[:300]}...")

        # Generate using the lmunit model (which returns a list of dicts)
        outputs = self.model.generate(prompt, self.sampling_params)

        if self.debug:
            print(f"    [DEBUG] Raw outputs: {outputs}")

        # LMUnit.generate() returns a list of dicts with 'score' and 'output_text'
        result = outputs[0] if isinstance(outputs, list) else outputs

        score_value = result.get('score', 0.0)
        output_text = result.get('output_text', '')

        if self.debug:
            print(f"    [DEBUG] Score: {score_value}, Output: {output_text[:100]}...")

        return LMUnitScore(
            unit_test=unit_test,
            score=score_value,
            token_probs={},  # LMUnit handles this internally
        )


def evaluate_file(
    evaluator: LMUnitEvaluator,
    results: dict,
    num_samples: Optional[int] = None,
    unit_tests: list[str] = None,
) -> list[EvaluationResult]:
    """
    Evaluate all responses in a results file.
    """
    if unit_tests is None:
        unit_tests = UNIT_TESTS
    
    items = results.get('results', [])
    
    if num_samples:
        items = items[:num_samples]
    
    print(f"Evaluating {len(items)} responses with {len(unit_tests)} unit tests each...")
    
    evaluation_results = []
    
    for i, item in enumerate(items):
        idx = item.get('idx', i)
        prompt = item['prompt']
        response = item['best_response']
        
        print(f"\n  [{i+1}/{len(items)}] Evaluating idx={idx}")
        
        scores = []
        for ut in unit_tests:
            score = evaluator.score(prompt, response, ut)
            scores.append(score)
            print(f"    {ut[:40]}... score={score.score:.2f}")
        
        avg_score = np.mean([s.score for s in scores])
        print(f"    => Avg score: {avg_score:.2f}")
        
        evaluation_results.append(EvaluationResult(
            idx=idx,
            prompt=prompt,
            response=response,
            scores=scores,
            avg_score=avg_score,
        ))
    
    return evaluation_results


def compute_statistics(results: list[EvaluationResult], unit_tests: list[str] = None) -> dict:
    """Compute aggregate statistics."""
    if unit_tests is None:
        unit_tests = UNIT_TESTS
    
    all_avg_scores = [r.avg_score for r in results]
    
    # Per-unit-test stats
    unit_test_stats = {}
    for i, ut in enumerate(unit_tests):
        scores = [r.scores[i].score for r in results if i < len(r.scores)]
        unit_test_stats[ut] = {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'min': np.min(scores) if scores else 0,
            'max': np.max(scores) if scores else 0,
        }
    
    return {
        'num_samples': len(results),
        'mean_score': np.mean(all_avg_scores) if all_avg_scores else 0,
        'std_score': np.std(all_avg_scores) if all_avg_scores else 0,
        'min_score': np.min(all_avg_scores) if all_avg_scores else 0,
        'max_score': np.max(all_avg_scores) if all_avg_scores else 0,
        'unit_test_stats': unit_test_stats,
    }


def print_results(stats: dict, label: str, results: list[EvaluationResult]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("LMUNIT EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nDataset: {label}")
    print(f"Samples evaluated: {stats['num_samples']}")
    
    print("\n" + "-" * 50)
    print("OVERALL SCORES (1-5 scale)")
    print("-" * 50)
    print(f"  Mean:  {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
    print(f"  Range: [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
    
    print("\n" + "-" * 50)
    print("PER-UNIT-TEST BREAKDOWN")
    print("-" * 50)
    
    for ut, ut_stats in stats['unit_test_stats'].items():
        print(f"\n  {ut[:60]}")
        print(f"    Mean: {ut_stats['mean']:.3f} ± {ut_stats['std']:.3f}  |  Range: [{ut_stats['min']:.3f}, {ut_stats['max']:.3f}]")
    
    print("\n" + "-" * 50)
    print("SAMPLE RESULTS")
    print("-" * 50)
    
    for r in results[:5]:
        print(f"\n  idx={r.idx}: {r.prompt[:60]}...")
        print(f"    Avg score: {r.avg_score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="LMUnit Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python lmunit_eval.py --file results.json \\
        --model ContextualAI/LMUnit-qwen2.5-72b --tp_size 4
        
    # With debug output
    python lmunit_eval.py --file results.json \\
        --model ContextualAI/LMUnit-qwen2.5-72b --tp_size 4 \\
        --debug --num_samples 2
"""
    )
    
    parser.add_argument("--file", type=str, required=True,
                        help="Path to results JSON file")
    parser.add_argument("--model", type=str, default="ContextualAI/LMUnit-qwen2.5-72b",
                        help="LMUnit model path")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="Tensor parallelism size")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--unit_tests", type=str, nargs="+", default=None,
                        help="Custom unit tests")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading {args.file}...")
    results = load_results(args.file)
    
    label = results.get('method', args.file)
    
    # Unit tests
    unit_tests = args.unit_tests if args.unit_tests else UNIT_TESTS
    print(f"\nUsing {len(unit_tests)} unit tests:")
    for i, ut in enumerate(unit_tests, 1):
        print(f"  {i}. {ut}")
    
    # Initialize evaluator
    print(f"\nInitializing LMUnit evaluator...")
    evaluator = LMUnitEvaluator(
        model_path=args.model,
        tp_size=args.tp_size,
        debug=args.debug
    )
    
    # Run evaluation
    eval_results = evaluate_file(
        evaluator=evaluator,
        results=results,
        num_samples=args.num_samples,
        unit_tests=unit_tests,
    )
    
    # Compute and print statistics
    stats = compute_statistics(eval_results, unit_tests=unit_tests)
    print_results(stats, label, eval_results)
    
    # Save results
    if args.output:
        output_data = {
            'config': {
                'file': args.file,
                'model': args.model,
                'unit_tests': unit_tests,
            },
            'statistics': {k: v for k, v in stats.items() if k != 'unit_test_stats'},
            'unit_test_statistics': stats['unit_test_stats'],
            'results': [
                {
                    'idx': r.idx,
                    'prompt': r.prompt,
                    'response': r.response,
                    'scores': {s.unit_test: s.score for s in r.scores},
                    'avg_score': r.avg_score,
                }
                for r in eval_results
            ],
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()