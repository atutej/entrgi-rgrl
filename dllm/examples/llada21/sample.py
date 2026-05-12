"""
python -u examples/llada21/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "inclusionAI/LLaDA2.1-mini"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.llada21.LLaDA21SamplerConfig):
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    threshold: float = 0.95
    editing_threshold: float = 0.9
    max_post_steps: int = 16
    num_to_transfer: int = 1


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.llada21.LLaDA21Sampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# Single prompt (batch size must be 1 due to per-sequence confidence-based progress)
messages = [
    [
        {
            "role": "user",
            "content": "Give a concise summary of diffusion-based text generation.",
        }
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

print("\n" + "=" * 80)
print("TEST: llada21.block_diffusion_generate()".center(80))
print("=" * 80)
for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize and outputs.histories is not None:
    terminal_visualizer.visualize(outputs.histories, rich=True)
