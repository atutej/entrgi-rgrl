"""Skywork reward model for open-ended generation tasks (e.g. WildChat)."""

import torch

_REWARD_MODEL = None
_REWARD_TOKENIZER = None
_LOADED_MODEL_NAME = None


def _get_reward_model(model_name: str):
    global _REWARD_MODEL, _REWARD_TOKENIZER, _LOADED_MODEL_NAME
    if _REWARD_MODEL is None or _LOADED_MODEL_NAME != model_name:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _REWARD_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _REWARD_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            num_labels=1,
        ).cuda()
        _REWARD_MODEL.eval()
        _LOADED_MODEL_NAME = model_name
    return _REWARD_MODEL, _REWARD_TOKENIZER


def make_skywork_reward_func(
    model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    max_length: int = 4096,
):
    """Return a GRPO-compatible reward function backed by the Skywork reward model.

    The returned function is a plain callable (prompts, completions) -> list[float]
    that lazily loads the reward model on first call and caches it for subsequent calls.
    """

    def skywork_reward_func(prompts, completions, **kwargs) -> list[float]:
        rm, tok = _get_reward_model(model_name)

        # prompts is a list of chat message lists; extract the last user turn.
        # completions is a list of [{"role": "assistant", "content": ...}].
        prompt_texts = [
            p[-1]["content"] if isinstance(p, list) else p for p in prompts
        ]
        completion_texts = [
            c[0]["content"] if isinstance(c, list) else c for c in completions
        ]

        convs = [
            [
                {"role": "user", "content": pt},
                {"role": "assistant", "content": ct},
            ]
            for pt, ct in zip(prompt_texts, completion_texts)
        ]

        texts = [
            tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in convs
        ]

        with torch.no_grad():
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(rm.device)
            scores = rm(**enc).logits.squeeze(-1)

        return [float(s) for s in scores.tolist()]

    skywork_reward_func.__name__ = f"skywork_{model_name.split('/')[-1]}"
    return skywork_reward_func
