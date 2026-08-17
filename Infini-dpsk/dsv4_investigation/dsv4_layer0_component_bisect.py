"""Isolate attention versus FFN parity in DSV4 deterministic layer zero."""

import ctypes
import gc
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

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
precision.NUM_LAYERS = 1
base_tiny_config = precision.tiny_config


def one_layer_config():
    config = base_tiny_config()
    config["num_hidden_layers"] = 1
    config["compress_ratios"] = [0]
    return config


precision.tiny_config = one_layer_config
variants = {
    "ffn_zero": (False, True, False, False, False),
    "routed_only": (True, False, False, True, False),
    "shared_only": (True, False, True, False, False),
    "ffn_both": (True, False, False, False, False),
    "routed_uniform": (True, False, False, True, True),
}

for name, (zero_attention, zero_ffn, zero_routed, zero_shared, zero_gate) in variants.items():
    model_dir = Path(f"/root/autodl-tmp/dpv4-layer0-{name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    config = one_layer_config()
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    state = precision.deterministic_state_dict(config)
    for key, tensor in list(state.items()):
        if zero_attention and key.startswith("layers.0.attn."):
            state[key] = torch.zeros_like(tensor)
        if zero_ffn and key.startswith("layers.0.ffn.") and tensor.is_floating_point():
            state[key] = torch.zeros_like(tensor)
        if zero_routed and key.startswith("layers.0.ffn.experts."):
            state[key] = torch.zeros_like(tensor)
        if zero_shared and key.startswith("layers.0.ffn.shared_experts."):
            state[key] = torch.zeros_like(tensor)
        if zero_gate and key == "layers.0.ffn.gate.weight":
            state[key] = torch.zeros_like(tensor)
    save_file(state, str(model_dir / "model.safetensors"))

    py_model = precision.load_python_reference(model_dir, "cuda")
    cpp_model = precision.load_cpp_model(model_dir, "cuda")
    input_ids = [3, 5, 7, 11, 13]
    with torch.inference_mode():
        py_logits = py_model(torch.tensor([input_ids], device="cuda")).logits.float().cpu().numpy()
    cpp_logits = precision.raw_cpp_forward(cpp_model, input_ids)
    diff = np.abs(cpp_logits - py_logits)
    print(name, {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "python_argmax": int(py_logits[0, -1].argmax()),
        "cpp_argmax": int(cpp_logits[0, -1].argmax()),
    })
    del cpp_model, py_model
    gc.collect()
    torch.cuda.empty_cache()
