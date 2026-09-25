from typing import (
    List,
    Any
)
import os
import pickle

import logging
import time

from functools import partial
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _call_api(func, limit=5, pause=10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                raise e
    return output

def key_of_llm_query(*args, **kwargs) -> str:
    sorted_kwargs = sorted(kwargs.items())
    key = pickle.dumps((args, tuple(sorted_kwargs)))
    return key

# NOTE: ONLY return 1 output per prompt
def _batch_openai_query(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None, # not needed for openai
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
):
    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]
        
    # TODO: not actually batching yet
    import openai
    if model_name == "deepseek-chat":
        client = openai.OpenAI(base_url="https://api.deepseek.com")
    else:
        client = openai.OpenAI()

    canonical_outputs = []
    for prompt in tqdm(prompts):
        func = partial(
            client.chat.completions.create, 
            model=model_name, 
            messages=prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **query_kwargs
        )
        completion = _call_api(func)
        content = completion.choices[0].message.content
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": content, "success": True,})
    return canonical_outputs

def _batch_together_query(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None, # not needed for openai
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
):

    # TODO: not actually batching yet
    import together
    client = together.Together()

    model_name = model_name.replace("together/", "")

    canonical_outputs = []
    for prompt in tqdm(prompts):
        func = partial(
            client.chat.completions.create, 
            model=model_name, 
            messages=prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **query_kwargs
        )
        completion = _call_api(func)
        content = completion.choices[0].message.content
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": content, "success": True,})
    return canonical_outputs

def _batch_gemini_query(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None, # not needed for gemini
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
):
    # Use OpenAI-compatible endpoint for Gemini
    import openai

    model_name = model_name.replace("gemini/", "")

    api_key = os.environ.get('GEMINI_API_KEY')
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    canonical_outputs = []
    for prompt in tqdm(prompts):
        func = partial(
            client.chat.completions.create,
            model=model_name,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **query_kwargs,
        )
        try:
            completion = _call_api(func)
            content = completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            content = ""
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": content, "success": True,})
    return canonical_outputs

def _batch_anthropic_query(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None, # not needed for anthropic
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
):
    import anthropic
    
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/"):]
    
    client = anthropic.Anthropic()

    canonical_outputs = []
    for prompt in tqdm(prompts):
        func = partial(
            client.messages.create,
            model=model_name,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **query_kwargs
        )
        completion = _call_api(func)
        content = completion.content[0].text
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": content, "success": True,})
    return canonical_outputs

class _VLLMBackend:
    llm = None
    tokenizer = None
    init_args = None # model_name, max_model_len

# NOTE: ONLY return 1 output per prompt
def _batch_vllm_query(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None,
    query_kwargs: dict = {}, # TODO: not using query_kwargs yet
    aux_kwargs: dict = {},
):
    assert model_name.startswith("vllm/")
    assert max_model_len is not None
    model_name = model_name[len("vllm/"):]

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    if _VLLMBackend.llm is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
        _VLLMBackend.llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",  max_model_len=max_model_len, gpu_memory_utilization=aux_kwargs.get("vllm_gpu_memory_utilization", 0.95))
        _VLLMBackend.init_args = (model_name, max_model_len)
        _VLLMBackend.tokenizer = tokenizer

    assert _VLLMBackend.init_args == (model_name, max_model_len)
    llm = _VLLMBackend.llm
    sampling_kwargs = {}
    if "stop" in query_kwargs:
        sampling_kwargs["stop"] = query_kwargs["stop"]  # NOTE: this is a list of strings
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, **sampling_kwargs)
    tokenizer = _VLLMBackend.tokenizer


    token_prompts = []
    # NOTE: pass in token ids which we find returns better results closer to hf generate
    for prompt in prompts:
        if "gpt-oss" in model_name and "reasoning_effort" in query_kwargs:
            reasoning_effort = query_kwargs["reasoning_effort"]
            assert reasoning_effort in ["low", "medium", "high"]
            token_prompts.append(TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(conversation=prompt, add_generation_prompt=True, reasoning_effort=reasoning_effort, tokenize=True)))
        else:
            token_prompts.append(TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(conversation=prompt, add_generation_prompt=True, tokenize=True,)))

    outputs = llm.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    canonical_outputs = []
    for prompt, output in zip(prompts, outputs):
        if "gpt-oss" in model_name:
            # remove the thinking part if it exists
            generated_token_ids = output.outputs[0].token_ids
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            if "<|start|>assistant<|channel|>final<|message|>" in generated_text:
                analysis_part, output_part = generated_text.split("<|start|>assistant<|channel|>final<|message|>", 1)
                analysis_part = analysis_part.replace("<|channel|>analysis<|message|>", "").replace("<|end|>", "").replace("<|return|>", "").strip()
                output_part = output_part.replace("<|return|>", "").replace("<|end|>", "").strip()
            else:
                output_part = output.outputs[0].text

            canonical_outputs.append({"model": model_name, "prompt": prompt, "output": output_part, "raw_output": generated_text, "success": True,})
        else:
            generated_text = output.outputs[0].text
            canonical_outputs.append({"model": model_name, "prompt": prompt, "output": generated_text, "success": True,})
    return canonical_outputs

def _batch_vllm_query_force_think(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    start_think_marker: str,
    top_p: float = 1.0,
    max_model_len: int = None,
    query_kwargs: dict = {}, 
    aux_kwargs: dict = {},
):
    assert model_name.startswith("vllm/")
    assert max_model_len is not None
    model_name = model_name[len("vllm/"):]

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    if _VLLMBackend.llm is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
        _VLLMBackend.llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",  max_model_len=max_model_len, gpu_memory_utilization=aux_kwargs.get("vllm_gpu_memory_utilization", 0.95))
        _VLLMBackend.init_args = (model_name, max_model_len)
        _VLLMBackend.tokenizer = tokenizer

    assert _VLLMBackend.init_args == (model_name, max_model_len)
    llm = _VLLMBackend.llm
    sampling_kwargs = {}
    if "stop" in query_kwargs:
        sampling_kwargs["stop"] = query_kwargs["stop"]  
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, **sampling_kwargs)
    tokenizer = _VLLMBackend.tokenizer

    token_prompts = []
    # NOTE: pass in token ids which we find returns better results closer to hf generate
    for prompt in prompts:
        generation_prompt = tokenizer.apply_chat_template(
            conversation=prompt, 
            add_generation_prompt=True, 
            tokenize=False,
        )
        generation_prompt = generation_prompt + start_think_marker
        tokenized = tokenizer.encode(generation_prompt)
        token_prompts.append(
            TokensPrompt(
                prompt_token_ids=tokenized
            )
        )

    outputs = llm.generate(
        prompts=token_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    canonical_outputs = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": generated_text, "success": True,})
    return canonical_outputs

## OBSOLETE: using token ids gives better results than using chat prompt
def _batch_vllm_query_chat_prompt(
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None,
    query_kwargs: dict = {}, # TODO: not using query_kwargs yet
    aux_kwargs: dict = {},
):
    assert model_name.startswith("vllm/")
    assert max_model_len is not None
    model_name = model_name[len("vllm/"):]

    import torch
    from vllm import LLM, SamplingParams

    if _VLLMBackend.llm is None:
        _VLLMBackend.llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",  max_model_len=max_model_len, gpu_memory_utilization=aux_kwargs.get("vllm_gpu_memory_utilization", 0.95))
        _VLLMBackend.init_args = (model_name, max_model_len)

    assert _VLLMBackend.init_args == (model_name, max_model_len)
    llm = _VLLMBackend.llm
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    outputs = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    

    canonical_outputs = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        canonical_outputs.append({"model": model_name, "prompt": prompt, "output": generated_text, "success": True,})
    return canonical_outputs


class _DiffusionBackend:
    """Singleton cache for a loaded dllm diffusion model, keyed by (backend_kind, model_name)."""
    model = None
    tokenizer = None
    sampler = None
    init_args = None  # (backend_kind, model_name)


def _load_diffusion_backend(backend_kind: str, model_name: str):
    assert backend_kind in ("dream", "llada")
    if _DiffusionBackend.init_args != (backend_kind, model_name):
        import dllm
        from dllm.utils.configs import ModelArguments

        model_args = ModelArguments(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

        if backend_kind == "dream":
            from dllm.pipelines import dream
            sampler = dream.DreamSampler(model=model, tokenizer=tokenizer)
        else:
            sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

        _DiffusionBackend.model = model
        _DiffusionBackend.tokenizer = tokenizer
        _DiffusionBackend.sampler = sampler
        _DiffusionBackend.init_args = (backend_kind, model_name)

    return _DiffusionBackend.model, _DiffusionBackend.tokenizer, _DiffusionBackend.sampler


# NOTE: ONLY return 1 output per prompt. aux_kwargs supports diffusion-specific knobs:
# steps, diffusion_batch_size (all); alg, alg_temp, top_k (dream); block_size, remasking (llada).
def _batch_diffusion_query(
    backend_kind: str,
    prompts: List[Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len: int = None,  # unused: diffusion samplers here have no separate context cap
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
):
    prefix = f"{backend_kind}/"
    assert model_name.startswith(prefix)
    model_name = model_name[len(prefix):]

    import dllm

    model, tokenizer, sampler = _load_diffusion_backend(backend_kind, model_name)

    if backend_kind == "dream":
        from dllm.pipelines import dream
        sampler_config = dream.DreamSamplerConfig(
            max_new_tokens=max_tokens,
            steps=aux_kwargs.get("steps", max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=aux_kwargs.get("top_k", 50),
            alg=aux_kwargs.get("alg", "entropy"),
            alg_temp=aux_kwargs.get("alg_temp", 0.0),
        )
    else:
        sampler_config = dllm.core.samplers.MDLMSamplerConfig(
            max_new_tokens=max_tokens,
            block_size=aux_kwargs.get("block_size", max_tokens),
            steps=aux_kwargs.get("steps", max_tokens),
            temperature=temperature,
            remasking=aux_kwargs.get("remasking", "low_confidence"),
        )

    batch_size = aux_kwargs.get("diffusion_batch_size", 8)
    canonical_outputs = []
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start : start + batch_size]
        input_ids = tokenizer.apply_chat_template(
            batch_prompts, add_generation_prompt=True, tokenize=True,
        )
        outputs = sampler.sample(input_ids, sampler_config, return_dict=True)
        texts = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), input_ids)
        for prompt, text in zip(batch_prompts, texts):
            canonical_outputs.append(
                {"model": f"{backend_kind}/{model_name}", "prompt": prompt, "output": text, "success": True}
            )
    return canonical_outputs


def _batch_dream_query(prompts, model_name, max_tokens, temperature, top_p=1.0, max_model_len=None, query_kwargs={}, aux_kwargs={}):
    return _batch_diffusion_query("dream", prompts, model_name, max_tokens, temperature, top_p, max_model_len, query_kwargs, aux_kwargs)


def _batch_llada_query(prompts, model_name, max_tokens, temperature, top_p=1.0, max_model_len=None, query_kwargs={}, aux_kwargs={}):
    return _batch_diffusion_query("llada", prompts, model_name, max_tokens, temperature, top_p, max_model_len, query_kwargs, aux_kwargs)


def batch_query(
    prompts: List[Any],
    model_name: str,
    gen_max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    max_model_len = None,
    query_kwargs: dict = {},
    aux_kwargs: dict = {},
    start_think_marker: str = None,
):
    # if start_think_marker is not None:
    #     assert model_name.startswith("vllm/"), "start_think_marker only supported for vllm models"
    #     outputs = _batch_vllm_query_force_think(prompts, model_name, gen_max_tokens, temperature, start_think_marker, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    if model_name.startswith("gpt") or model_name.startswith("openai/"):
        outputs = _batch_openai_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("gemini"):
        outputs = _batch_gemini_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("anthropic/"):
        outputs = _batch_anthropic_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("together/"):
        outputs = _batch_together_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("vllm/"):
        outputs = _batch_vllm_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("dream/"):
        outputs = _batch_dream_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    elif model_name.startswith("llada/"):
        outputs = _batch_llada_query(prompts, model_name, gen_max_tokens, temperature, top_p, max_model_len, query_kwargs=query_kwargs, aux_kwargs=aux_kwargs)
    else:
        raise NotImplementedError

    assert len(outputs) == len(prompts)
    assert "model" in outputs[0] and "prompt" in outputs[0] and "output" in outputs[0] and "success" in outputs[0]

    return outputs

