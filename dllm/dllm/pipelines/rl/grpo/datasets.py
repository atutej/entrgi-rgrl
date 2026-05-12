"""
Dataset loaders and reward-function mappings for RL (GRPO) training.

Each supported task returns a (Dataset, reward_functions) pair via
`get_dataset_and_rewards`.  Keeping this here (rather than in dllm/data)
means the RL-specific prompt construction and reward wiring stays together
with the RL pipeline, while example scripts stay thin.
"""

import os
from typing import Callable, Optional

from datasets import Dataset, load_dataset

from dllm.pipelines.rl.grpo.rewards.code import coding_reward_func
from dllm.pipelines.rl.grpo.rewards.skywork import make_skywork_reward_func
from dllm.pipelines.rl.grpo.rewards.countdown import countdown_reward_func
from dllm.pipelines.rl.grpo.rewards.format import (
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from dllm.pipelines.rl.grpo.rewards.math import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    int_reward_func,
)
from dllm.pipelines.rl.grpo.rewards.sudoku import sudoku_reward_func

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]}
            ],
            "answer": _extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic "
                        f"expression that evaluates to exactly {x['target']}. You must use all numbers "
                        "from the list, and each number must be used exactly once. You may use the "
                        "operations +, -, *, and / as needed. After reasoning, provide only your final "
                        "expression inside <answer></answer> tags without including an equals sign or "
                        "the target number. For example, if the numbers are [2, 3, 4] and the target "
                        "is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
                    ),
                }
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    import pandas as pd

    sudoku_file_path = os.environ.get(
        "SUDOKU_DATA_PATH",
        "d1/dataset/4x4_sudoku_unique_puzzles.csv",
    )
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                }
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to "
                        f"solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}"
                    ),
                }
            ],
            "answer": x["solution"],
        }
    )


def get_code_questions(split="train") -> Dataset:
    data = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)
    data = data.train_test_split(test_size=0.1, seed=42)["train"]
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\n\nYou are a coding expert. You will be given a coding problem "
                        f"to solve. Solve it step by step. \n\n{x['question']}"
                    ),
                }
            ],
            "answer": {"solution": x["solution"], "tests": x["test"]},
        }
    )


def get_magpie_ultra_questions(
    num_prompts: int = 20000,
    seed: int = 42,
    min_quality: str = "good",
) -> Dataset:
    quality_order = {"very poor": 0, "poor": 1, "average": 2, "good": 3, "excellent": 4}
    min_q = quality_order[min_quality]
    data = load_dataset("argilla/magpie-ultra-v0.1", split="train")
    data = data.filter(lambda x: quality_order.get(x["quality"], 0) >= min_q)
    data = data.shuffle(seed=seed)
    if num_prompts > 0:
        data = data.select(range(min(num_prompts, len(data))))
    return data.map(
        lambda x: {"prompt": [{"role": "user", "content": x["instruction"]}]},
        remove_columns=data.column_names,
    )


def get_lmsys_questions(
    num_prompts: int = 20000,
    seed: int = 42,
) -> Dataset:
    data = load_dataset("lmsys/lmsys-chat-1m", split="train")
    data = data.filter(lambda x: x["language"] == "English")
    data = data.shuffle(seed=seed)
    if num_prompts > 0:
        data = data.select(range(min(num_prompts, len(data))))
    return data.map(
        lambda x: {"prompt": [{"role": "user", "content": x["conversation"][0]["content"]}]},
        remove_columns=data.column_names,
    )


def get_wildchat_questions(
    num_prompts: int = 20000,
    seed: int = 42,
) -> Dataset:
    data = load_dataset("allenai/tulu-3-wildchat-if-on-policy-8b", split="train")
    data = data.shuffle(seed=seed)
    if num_prompts > 0:
        data = data.select(range(min(num_prompts, len(data))))

    def _extract_prompt(row):
        for key in ("prompt", "messages", "chosen", "rejected"):
            v = row.get(key)
            if not v:
                continue
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list):
                user_turns = [
                    m.get("content")
                    for m in v
                    if isinstance(m, dict)
                    and m.get("role") == "user"
                    and m.get("content")
                ]
                if user_turns:
                    return user_turns[-1].strip()
        return None

    rows = []
    seen = set()
    for row in data:
        t = _extract_prompt(row)
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append({"prompt": [{"role": "user", "content": t}]})

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ("gsm8k", "countdown", "sudoku", "math", "code", "wildchat", "magpie", "lmsys")


def get_dataset_and_rewards(
    dataset_name: str,
    reward_model: Optional[str] = None,
) -> tuple[Dataset, list[Callable]]:
    """Return (dataset, reward_functions) for the given task name.

    Args:
        dataset_name: One of ``"gsm8k"``, ``"countdown"``, ``"sudoku"``,
            ``"math"``, ``"code"``, or ``"wildchat"``.
        reward_model: Override the reward model used for open-ended tasks
            (wildchat).  Ignored for verifiable-reward tasks.  Defaults to
            ``make_skywork_reward_func()``'s built-in default when ``None``.

    Returns:
        A ``(dataset, reward_functions)`` tuple ready to pass to
        ``DiffuGRPOTrainer``.  Reward functions have ``verbose=False`` by
        default; wrap with ``functools.partial(fn, verbose=True)`` if needed.
    """
    if dataset_name == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif dataset_name == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif dataset_name == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif dataset_name == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif dataset_name == "code":
        dataset = get_code_questions("train")
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    elif dataset_name == "wildchat":
        dataset = get_wildchat_questions()
        reward_functions = [
            make_skywork_reward_func(reward_model)
            if reward_model is not None
            else make_skywork_reward_func()
        ]
    elif dataset_name == "magpie":
        dataset = get_magpie_ultra_questions()
        reward_functions = [
            make_skywork_reward_func(reward_model)
            if reward_model is not None
            else make_skywork_reward_func()
        ]
    elif dataset_name == "lmsys":
        dataset = get_lmsys_questions()
        reward_functions = [
            make_skywork_reward_func(reward_model)
            if reward_model is not None
            else make_skywork_reward_func()
        ]
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        )

    return dataset, reward_functions
