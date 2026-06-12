"""
BERT-base fine-tune throughput benchmark.

Real BERT-base architecture (BertForSequenceClassification, default = bert-base
dims) instantiated from config — no pretrained download. Synthetic batch of
random token ids; we measure compute throughput, not dataset I/O. Returns
sequences/sec over a measure window after warmup.

Honors batch_size, precision, compile_mode, use_flash_attention, and
gradient_accumulation_steps. dataloader_workers is inert here: the batch is
synthetic and resident on-device, exactly like the ResNet-50 workload, so there
is no host-side data pipeline for worker count to affect.
"""
from __future__ import annotations

import contextlib
import time

WARMUP_STEPS = 5
MEASURE_STEPS = 20
SEQ_LEN = 128
VOCAB_SIZE = 30522  # bert-base-uncased
NUM_LABELS = 2

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
        # torch < 2.1: legacy global SDPA switch.
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=use_flash,
                enable_mem_efficient=use_flash,
                enable_math=True,
            )
        except Exception:
            return contextlib.nullcontext()


def _build_model(model_cfg):
    from transformers import BertForSequenceClassification

    # Prefer the explicit attn_implementation route so the SDPA toggle has teeth;
    # fall back to a plain constructor on older transformers.
    try:
        return BertForSequenceClassification._from_config(
            model_cfg, attn_implementation="sdpa"
        )
    except (TypeError, ValueError):
        return BertForSequenceClassification(model_cfg)


def train(config: dict) -> float:
    import torch
    from transformers import BertConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the paper benchmarks")

    device = torch.device("cuda")
    batch_size = int(config["batch_size"])
    precision = config.get("precision", "fp32")
    dtype = getattr(torch, _DTYPES[precision])
    accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1) or 1))
    use_flash = bool(config.get("use_flash_attention", False))

    model_cfg = BertConfig(vocab_size=VOCAB_SIZE, num_labels=NUM_LABELS)
    model = _build_model(model_cfg).to(device)
    model.train()
    if config.get("compile_mode") and config["compile_mode"] != "none":
        model = torch.compile(model, mode=config["compile_mode"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device)
    attention_mask = torch.ones(batch_size, SEQ_LEN, dtype=torch.long, device=device)
    labels = torch.randint(0, NUM_LABELS, (batch_size,), device=device)
    use_autocast = precision != "fp32"

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    ).loss
            else:
                loss = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                ).loss
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
