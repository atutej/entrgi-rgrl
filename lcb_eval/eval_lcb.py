"""
LiveCodeBench evaluation for Dream-org/Dream-v0-Instruct-7B (or any dllm-loadable Dream
checkpoint), using the plain (unguided) DreamSampler.

Covers BOTH LCB test formats -- function-call-style problems (sourced from LeetCode, with a
`class Solution:` starter-code stub) and stdin/stdout-style competitive programming problems
(Codeforces/AtCoder, no starter code, reads from stdin/prints to stdout) -- with no platform
filtering, matching both (a) lcb_runner's own canonical scoring (aggregates pass@1 across all
platforms by default -- see compute_scores.py's `--platform` defaulting to None) and (b)
zhangyitonggg/dllm4code's LiveCodeBench script, which also generates for every problem with no
platform filter. build_prompt() branches on starter_code truthiness the same way that script does.

Scoring is delegated entirely to the OFFICIAL LiveCodeBench evaluator (lcb_runner, cloned into
../LiveCodeBench and `pip install -e --no-deps`'d into this conda env) rather than a hand-rolled
sandbox -- see score_lcb.py in this folder, which shells out to
`python -m lcb_runner.runner.custom_evaluator`. That evaluator already implements execution for
both test types (TestType.STDIN and TestType.FUNCTIONAL), so no format-specific code lives here.

"target" / --test_sample_size: by default every problem in the chosen --release_version/date range
is a target (real generation). --test_sample_size instead takes a random subset of that range as
targets, for a quick experiment; non-target problems still get an entry in the final output (empty
code_list) because lcb_runner's custom_evaluator reconstructs its own copy of the FULL benchmark
for the same --release_version/--start_date/--end_date and asserts
`len(custom_outputs) == len(benchmark)` -- so every problem in range needs an entry regardless of
whether we actually generated for it. score_lcb.py reports pass@1 restricted to the target subset
from the evaluator's own per-question results when --test_sample_size narrowed it, so a subsampled
run's number isn't diluted by problems that were never really attempted.

One process = one shard = one GPU (pin with CUDA_VISIBLE_DEVICES before launching; this script
itself never touches CUDA_VISIBLE_DEVICES). run_lcb_eval.sh launches --num_shards copies of this
script, each with a different --shard_index, then calls this same script again with --merge.

Decoding: same dllm.pipelines.dream.DreamSampler used throughout math_eval/math500_train --
Dream-Coder's own recommended generation config uses the identical alg/steps/temperature/top_p
interface (confirmed against its model card), so this same sampler is expected to work unchanged
if a Dream-Coder checkpoint is passed via --model later; no guidance seam is wired in yet (unlike
math_eval/eval_math500.py) since this is the initial experiment -- add one the same way if needed.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# LiveCodeBench/ is the official repo (git clone https://github.com/LiveCodeBench/LiveCodeBench),
# pip installed --no-deps into this conda env; we only import its dataclasses/enums here, never
# its runner (that stays a separate subprocess in score_lcb.py so a scoring-side crash can't take
# down a generation run, and so this script doesn't need lcb_runner's own heavier deps).
LCB_DIR = Path(__file__).resolve().parent.parent / "LiveCodeBench"
sys.path.insert(0, str(LCB_DIR))
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # noqa: E402

import dllm  # noqa: E402
from dllm.pipelines import dream  # noqa: E402
from dllm.utils.configs import ModelArguments  # noqa: E402

ORIG_CWD = os.getcwd()

LCB_REPO_ID = "livecodebench/code_generation_lite"
# Mirrors LiveCodeBench/'s own code_generation_lite.py _VERSIONS_CONFIGS mapping -- release_vN
# cumulatively includes test.jsonl..testN.jsonl.
_ALL_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
VERSION_FILES = {f"release_v{i}": _ALL_FILES[:i] for i in range(1, 7)}
VERSION_FILES["release_latest"] = _ALL_FILES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B",
                    help="HF repo id or local checkpoint dir, loaded via dllm.utils.get_model.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)

    p.add_argument("--release_version", type=str, default="release_v1",
                    help="release_v1..release_v6 or release_latest. release_v1 (400 problems, "
                    "181 LeetCode/functional) is the smallest, used as the initial-experiment default.")
    p.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD, inclusive.")
    p.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD, inclusive.")

    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=None, help="Defaults to --max_new_tokens.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--alg", type=str, default="entropy",
                    choices=["maskgit_plus", "topk_margin", "entropy"])
    p.add_argument("--alg_temp", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=8, help="Prompts per forward-pass batch.")

    p.add_argument("--test_sample_size", type=int, default=None,
                    help="Only generate for this many of the LeetCode/functional (target) "
                    "problems (random subset, before sharding). Default: all of them.")

    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--force_overwrite", action="store_true", default=False)
    p.add_argument("--merge", action="store_true", default=False)

    args = p.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(ORIG_CWD, args.output_dir)
    local_model_path = os.path.join(ORIG_CWD, args.model)
    if not os.path.isabs(args.model) and os.path.isdir(local_model_path):
        args.model = local_model_path

    return args


def model_saving_name(model_path: str) -> str:
    return model_path.rstrip("/").replace("/", "-")


def shard_output_path(args) -> str:
    name = model_saving_name(args.model)
    steps = args.steps or args.max_new_tokens
    tag = (f"lcb_{args.release_version}_{name}_max{args.max_new_tokens}steps{steps}"
           f"t{args.temperature}p{args.top_p}_{args.seed}")
    suffix = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    return os.path.join(args.output_dir, f"{tag}{suffix}.json")


def merged_output_path(args) -> str:
    merged_args = argparse.Namespace(**vars(args))
    merged_args.num_shards = 1
    merged_args.shard_index = 0
    return shard_output_path(merged_args)


def load_lcb_dataset(release_version, start_date=None, end_date=None):
    """Downloads the raw .jsonl files directly via huggingface_hub (NOT datasets.load_dataset --
    that requires the dataset's own loading SCRIPT, which current `datasets` versions refuse to
    run at all -- "Dataset scripts are no longer supported"). CodeGenerationProblem (from the
    installed lcb_runner package) does the actual field decoding (including the base64/zlib/
    pickle-compressed private_test_cases), identically to what the script would have done."""
    from huggingface_hub import hf_hub_download

    files = VERSION_FILES[release_version]
    problems = []
    for fname in files:
        path = hf_hub_download(LCB_REPO_ID, fname, repo_type="dataset")
        with open(path) as f:
            for line in f:
                problems.append(CodeGenerationProblem(**json.loads(line)))

    if start_date is not None:
        d = datetime.strptime(start_date, "%Y-%m-%d")
        problems = [p for p in problems if d <= p.contest_date.replace(tzinfo=None)]
    if end_date is not None:
        d = datetime.strptime(end_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date.replace(tzinfo=None) <= d]

    # Matches lcb_runner.runner.scenario_router.build_prompt_benchmark's own sort -- the
    # evaluator reconstructs its benchmark this same way, so this ordering isn't load-bearing for
    # correctness (question_id is the join key either side uses), but keeping it identical avoids
    # any doubt when diffing.
    problems.sort(key=lambda x: x.question_id)
    return problems


def build_prompt(problem: CodeGenerationProblem):
    content = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        content += (
            "You will use the following starter code to write the solution to the problem, "
            "and enclose your code within delimiters as follows.\n"
            f"```python\n{problem.starter_code}\n```\n\n"
        )
    else:
        content += (
            "Read the inputs from stdin and print the output to stdout. Enclose your code "
            "within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n\n"
        )
    content += "Return your complete solution wrapped in a single python code block."
    return [{"role": "user", "content": content}]


def extract_code(model_output: str) -> str:
    """Same convention as LiveCodeBench/lcb_runner/utils/extraction_utils.py's own extract_code
    generic-model branch: content between the LAST two ``` fence markers. Returns "" (matching
    their own behavior) if fewer than two fences are found."""
    lines = model_output.split("\n")
    fence_idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_idx) < 2:
        return ""
    return "\n".join(lines[fence_idx[-2] + 1: fence_idx[-1]])


def load_shard(args):
    problems = load_lcb_dataset(args.release_version, args.start_date, args.end_date)

    # All platforms/test types are targets (matches both lcb_runner's own canonical scoring,
    # which aggregates across platforms by default, and dllm4code's LiveCodeBench script, which
    # generates for every problem with no platform filter -- build_prompt() already branches on
    # starter_code truthiness the same way theirs does, so stdin/stdout problems get a real
    # generation attempt too, not just a placeholder).
    all_ids = [p.question_id for p in problems]
    if args.test_sample_size is not None:
        rng = random.Random(args.seed)
        target_ids = set(rng.sample(all_ids, min(args.test_sample_size, len(all_ids))))
    else:
        target_ids = set(all_ids)

    for p in problems:
        p.is_target = p.question_id in target_ids  # dataclass allows attribute assignment

    return problems[args.shard_index::args.num_shards]


def run_generation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = shard_output_path(args)
    if os.path.exists(out_path) and not args.force_overwrite:
        print(f"{out_path} already exists, skipping (use --force_overwrite to redo).")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shard = load_shard(args)
    target_shard = [p for p in shard if p.is_target]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard)} total problems, "
          f"{len(target_shard)} target (LeetCode/functional) will actually be generated, "
          f"model={args.model}")

    model_args = ModelArguments(model_name_or_path=args.model, dtype="bfloat16")
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    sampler = dream.DreamSampler(model=model, tokenizer=tokenizer)
    sampler_config = dream.DreamSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps or args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )

    generated = {}
    for start in tqdm(range(0, len(target_shard), args.batch_size), desc=f"shard {args.shard_index}"):
        batch = target_shard[start:start + args.batch_size]
        chats = [build_prompt(p) for p in batch]
        input_ids = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)

        outputs = sampler.sample(input_ids, sampler_config, return_dict=True)
        texts = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), input_ids)

        for p, text in zip(batch, texts):
            generated[p.question_id] = text

    results = []
    for p in shard:
        raw_text = generated.get(p.question_id, "")
        results.append({
            "question_id": p.question_id,
            "platform": p.platform.value,
            "difficulty": p.difficulty.value,
            "is_target": p.is_target,
            "raw_completion": raw_text,
            "code": extract_code(raw_text) if raw_text else "",
        })

    n_correct_extract = sum(1 for r in results if r["is_target"] and r["code"])
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Shard {args.shard_index}/{args.num_shards}: {n_correct_extract}/{len(target_shard)} "
          f"target completions had an extractable code block -> {out_path}")


def run_merge(args):
    all_results = []
    for shard_index in range(args.num_shards):
        shard_args = argparse.Namespace(**vars(args))
        shard_args.shard_index = shard_index
        with open(shard_output_path(shard_args)) as f:
            all_results.extend(json.load(f)["results"])

    out_path = merged_output_path(args)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    # lcb_runner's custom_evaluator asserts len(custom_outputs) == len(its own reconstructed
    # benchmark for this release_version/date range) -- so every problem needs an entry here,
    # including non-target ones (empty code_list, harmlessly fails their own tests).
    lcb_submission = [{"question_id": r["question_id"], "code_list": [r["code"]]} for r in all_results]
    lcb_path = out_path.replace(".json", "_lcb_submission.json")
    with open(lcb_path, "w") as f:
        json.dump(lcb_submission, f)

    n_target = sum(1 for r in all_results if r["is_target"])
    print(f"Merged {args.num_shards} shards ({len(all_results)} total problems, "
          f"{n_target} target/functional) -> {out_path}")
    print(f"lcb_runner submission file -> {lcb_path}")
    print(f"Next: python score_lcb.py --custom_output_file {lcb_path} "
          f"--release_version {args.release_version}"
          + (f" --start_date {args.start_date}" if args.start_date else "")
          + (f" --end_date {args.end_date}" if args.end_date else ""))


def main():
    args = parse_args()
    if args.merge:
        run_merge(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
