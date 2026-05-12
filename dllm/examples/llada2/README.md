# LLaDA2.0

> 📄 Paper: [Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745) | 💻 Code: [github.com/inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)

Resources and examples for sampling **LLaDA2.0**.

## Files
```
# Pipeline modules relevant to LLaDA2
dllm/pipelines/llada2
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_llada2_moe.py # LLaDA2-MoE model configuration
│   └── modeling_llada2_moe.py      # LLaDA2-MoE model architecture
└── sampler.py                      # Inference module

# Example entry points for inference
examples/llada2
├── chat.py      # Multi-turn chat demo (uses chat template)
├── sample.py    # Single-turn sampling demo
└── README.md    # Documentation (You are here)
```

## Inference
Set `--model_name_or_path` to your checkpoint (e.g., `inclusionAI/LLaDA2.0-mini`).

We support single-turn sampling with visualization:
```shell
python -u examples/llada2/sample.py --model_name_or_path "inclusionAI/LLaDA2.0-mini"
```

We also support interactive multi-turn dialogue:
```shell
python -u examples/llada2/chat.py --model_name_or_path "inclusionAI/LLaDA2.0-mini"
```
