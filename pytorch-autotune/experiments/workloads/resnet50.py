"""
ResNet-50 throughput benchmark. Synthetic data — measuring compute, not I/O.

Returns samples/sec over a measure window after warmup. Always call
torch.cuda.synchronize() at the timer bookends so we don't time queued kernels.
"""
from __future__ import annotations

import time

WARMUP_STEPS = 5
MEASURE_STEPS = 20


def train(config: dict) -> float:
    import torch
    from torchvision import models

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the paper benchmarks")

    device = torch.device("cuda")
    batch_size = int(config["batch_size"])
    precision = config.get("precision", "fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    model = models.resnet50(weights=None).to(device)
    if config.get("compile_mode") and config["compile_mode"] != "none":
        model = torch.compile(model, mode=config["compile_mode"])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    use_autocast = precision != "fp32"

    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = loss_fn(model(x), y)
        else:
            loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = loss_fn(model(x), y)
        else:
            loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * MEASURE_STEPS) / elapsed
