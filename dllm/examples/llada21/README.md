# LLaDA2.1

> 📄 Paper: [LLaDA2.1](https://arxiv.org/abs/2602.08676) | 💻 Code: [github.com/inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)

Resources and examples for sampling **LLaDA2.1**.

## Files
```
# Pipeline modules relevant to LLaDA2.1
dllm/pipelines/llada21
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_llada21_moe.py # LLaDA2.1-MoE model configuration
│   └── modeling_llada21_moe.py      # LLaDA2.1-MoE model architecture
└── sampler.py                      # Inference module

# Example entry points for inference
examples/llada21
├── chat.py      # Multi-turn chat demo (uses chat template)
├── sample.py    # Single-turn sampling demo
└── README.md    # Documentation (You are here)
```

## Inference
Set `--model_name_or_path` to your checkpoint (e.g., `inclusionAI/LLaDA2.1-mini`).

We support interactive multi-turn dialogue with visualization:
```shell
python -u examples/llada21/chat.py --model_name_or_path "inclusionAI/LLaDA2.1-mini"
```

We also support single-turn sampling with generation history visualization:
```shell
python -u examples/llada21/sample.py --model_name_or_path "inclusionAI/LLaDA2.1-mini"
```
