"""Exercise DSV4 aggregation with two fixed, equally weighted experts."""

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

num_experts = int(os.environ.get("DSV4_TEST_NUM_EXPERTS", "2"))
top_k = int(os.environ.get("DSV4_TEST_TOP_K", "2"))
num_hash_layers = int(os.environ.get("DSV4_TEST_HASH_LAYERS", "1"))
norm_topk_prob = os.environ.get("DSV4_TEST_NORM_TOPK", "1") == "1"

precision.NUM_LAYERS = 1
precision.N_EXPERTS = num_experts
precision.TOP_K = top_k
base_tiny_config = precision.tiny_config


def two_expert_config():
    config = base_tiny_config()
    config.update(
        {
            "num_hidden_layers": 1,
            "compress_ratios": [0],
            "n_routed_experts": num_experts,
            "num_experts_per_tok": top_k,
            "n_shared_experts": 0,
            "num_hash_layers": num_hash_layers,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": norm_topk_prob,
        }
    )
    return config


precision.tiny_config = two_expert_config
model_dir = Path(
    f"/root/autodl-tmp/dpv4-route-{num_experts}e-top{top_k}-hash{num_hash_layers}"
    f"-norm{int(norm_topk_prob)}"
)
model_dir.mkdir(parents=True, exist_ok=True)
config = two_expert_config()
(model_dir / "config.json").write_text(json.dumps(config, indent=2))
state = precision.deterministic_state_dict(config)

for key, tensor in list(state.items()):
    if key.startswith("layers.0.attn."):
        state[key] = torch.zeros_like(tensor)
    if ".shared_experts." in key:
        del state[key]
    if key == "layers.0.ffn.gate.weight":
        state[key] = torch.zeros_like(tensor)

save_file(state, str(model_dir / "model.safetensors"))

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
print(
    f"route_{num_experts}e_top{top_k}_hash{num_hash_layers}_norm{int(norm_topk_prob)}",
    {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "python_argmax": int(py_logits[0, -1].argmax()),
        "cpp_argmax": int(cpp_logits[0, -1].argmax()),
    },
)

del cpp_model, py_model
gc.collect()
torch.cuda.empty_cache()
