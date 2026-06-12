"""
LLaMA-7B LoRA fine-tune throughput benchmark.

LLaMA-7B architecture (LlamaForCausalLM at 7B dims) instantiated from config —
no pretrained download, synthetic weights. Base is frozen; PEFT LoRA adapters
are the only trainable params. Synthetic input ids; we measure compute
throughput, not dataset I/O. Returns sequences/sec over a measure window.

The base model is constructed directly in the trial's target dtype (via the
default-dtype context) so we never hold a ~28 GB fp32 copy on the host before
moving to GPU — bf16 lands ~14 GB, matching the VRAM estimates in
autotune.profile.

Honors batch_size, precision, compile_mode, use_flash_attention, and
gradient_accumulation_steps. dataloader_workers is inert (synthetic on-device
batch, no host data pipeline), as in the other workloads.
"""
from __future__ import annotations

import contextlib
import time

WARMUP_STEPS = 2
MEASURE_STEPS = 5
SEQ_LEN = 512
VOCAB_SIZE = 32000
MAX_POSITION = 4096

# LLaMA-7B architecture.
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 32

_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _sdpa_context(use_flash: bool):
    """Make use_flash_attention a real toggle by pinning the SDPA backend.

    FLASH is fp16/bf16-only; in fp32 it is simply ineligible and PyTorch falls
    back to the math kernel automatically, so precision needs no special-casing.
    """
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if use_flash:
            backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        else:
            backends = [SDPBackend.MATH]
        return sdpa_kernel(backends)
    except Exception:
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=use_flash,
                enable_mem_efficient=use_flash,
                enable_math=True,
            )
        except Exception:
            return contextlib.nullcontext()


def _build_base(model_cfg):
    from transformers import LlamaForCausalLM

    try:
        return LlamaForCausalLM._from_config(model_cfg, attn_implementation="sdpa")
    except (TypeError, ValueError):
        return LlamaForCausalLM(model_cfg)


def train(config: dict) -> float:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the paper benchmarks")

    device = torch.device("cuda")
    batch_size = int(config["batch_size"])
    precision = config.get("precision", "bf16")
    dtype = getattr(torch, _DTYPES[precision])
    accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1) or 1))
    use_flash = bool(config.get("use_flash_attention", False))

    model_cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_POSITION,
    )

    # Construct the frozen base directly in the target dtype to avoid a 28 GB
    # fp32 host allocation before the GPU move.
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        base = _build_base(model_cfg)
    finally:
        torch.set_default_dtype(prev_default)
    base = base.to(device)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.train()
    if config.get("compile_mode") and config["compile_mode"] != "none":
        model = torch.compile(model, mode=config["compile_mode"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device)
    use_autocast = precision != "fp32"

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = model(input_ids=input_ids, labels=input_ids).loss
            else:
                loss = model(input_ids=input_ids, labels=input_ids).loss
            (loss / accum_steps).backward()
        optimizer.step()

    with _sdpa_context(use_flash):
        for _ in range(WARMUP_STEPS):
            step()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (batch_size * accum_steps * MEASURE_STEPS) / elapsed
