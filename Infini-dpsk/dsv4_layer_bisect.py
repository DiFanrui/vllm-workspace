"""Bisect DSV4 Python/C++ parity by number of deterministic decoder layers."""

import ctypes
import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

import infinicore
import infinilm.generation.utils


TEST = Path(
    "/root/autodl-tmp/InfiniLM-dpv4-test/test/models/deepseek_v4/"
    "test_deepseek_v4_deterministic_precision.py"
)
spec = importlib.util.spec_from_file_location("dsv4_precision", TEST)
precision = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(precision)

original_to_numpy = infinicore.Tensor.to_numpy


def to_numpy(tensor):
    if tensor.dtype != infinicore.bfloat16:
        return original_to_numpy(tensor)
    cpu = tensor.to(infinicore.device("cpu", 0))
    bits = np.ctypeslib.as_array(
        (ctypes.c_uint16 * cpu.numel()).from_address(cpu.data_ptr())
    ).copy()
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(cpu.shape)


infinicore.Tensor.to_numpy = to_numpy
os.environ["DEEPSEEK_V4_CODE"] = (
    "/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash/code"
)
base_tiny_config = precision.tiny_config
patterns = {0: [], 1: [0], 2: [0, 4], 3: [0, 4, 0]}

for layer_count in range(4):
    precision.NUM_LAYERS = layer_count

    def variant_config(layer_count=layer_count):
        config = base_tiny_config()
        config["num_hidden_layers"] = layer_count
        config["compress_ratios"] = patterns[layer_count]
        return config

    precision.tiny_config = variant_config
    model_dir = precision.write_tiny_model(
        Path(f"/root/autodl-tmp/dpv4-layer-bisect-{layer_count}")
    )
    py_model = precision.load_python_reference(model_dir, "cuda")
    cpp_model = precision.load_cpp_model(model_dir, "cuda")
    input_ids = [3, 5, 7, 11, 13]
    with torch.inference_mode():
        py_logits = (
            py_model(torch.tensor([input_ids], device="cuda"))
            .logits.float().cpu().numpy()
        )
    cpp_logits = precision.raw_cpp_forward(cpp_model, input_ids)
    diff = np.abs(cpp_logits - py_logits)
    print(f"layers={layer_count}", {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "python_argmax": int(py_logits[0, -1].argmax()),
        "cpp_argmax": int(cpp_logits[0, -1].argmax()),
    })
    del cpp_model, py_model
    gc.collect()
    torch.cuda.empty_cache()
