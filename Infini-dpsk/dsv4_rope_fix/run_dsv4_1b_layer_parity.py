"""Parity for a single-layer (or full) DSV4 checkpoint: python ref vs C++.

Usage:
  python run_dsv4_1b_layer_parity.py /path/to/model_dir [zero_pos]

Set ZERO_POS=1 (or pass `zero_pos`) to force all position ids to 0 (RoPE identity)
as a control to isolate the rotary path.
"""

import ctypes
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


INPUT_IDS = [3, 5, 7, 11, 13]


def bf16_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    cpu = tensor.to(infinicore.device("cpu", 0))
    bits = np.ctypeslib.as_array(
        (ctypes.c_uint16 * cpu.numel()).from_address(cpu.data_ptr())
    ).copy()
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(cpu.shape)


def raw_cpp_forward(model: InferEngine, input_ids: list[int], zero_pos: bool) -> np.ndarray:
    raw = super(InferEngine, model)
    seq = len(input_ids)
    block = model.get_cache_config().block_size()
    max_blocks = (seq + 2 + block - 1) // block

    def u(tensor):
        return tensor._underlying if hasattr(tensor, "_underlying") else tensor

    pos = [0] * seq if zero_pos else list(range(seq))
    output = raw.forward(
        raw.Input(
            u(infinicore.from_list([input_ids], dtype=infinicore.int64).view([1, seq])),
            position_ids=u(infinicore.from_list(pos, dtype=infinicore.int64)),
            past_sequence_lengths=u(infinicore.from_list([0], dtype=infinicore.int32)),
            total_sequence_lengths=u(infinicore.from_list([seq], dtype=infinicore.int32)),
            input_offsets=u(infinicore.from_list([0, seq], dtype=infinicore.int32)),
            cu_seqlens=u(infinicore.from_list([0, seq], dtype=infinicore.int32)),
            block_tables=u(
                infinicore.from_list([list(range(max_blocks))], dtype=infinicore.int32)
            ),
            slot_mapping=u(infinicore.from_list(list(range(seq)), dtype=infinicore.int64)),
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )
    )
    return bf16_to_numpy(infinicore.Tensor(output.logits))


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash"
    )
    zero_pos = (len(sys.argv) > 2 and sys.argv[2] == "zero_pos") or \
               __import__("os").environ.get("ZERO_POS", "") == "1"

    sys.path.insert(0, str(model_dir / "code"))
    from deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    py_config = DeepseekV4Config.from_pretrained(model_dir)
    py_model = DeepseekV4ForCausalLM.from_pretrained(
        model_dir, config=py_config, torch_dtype=torch.bfloat16
    ).cuda().eval()
    with torch.inference_mode():
        py_logits = py_model(torch.tensor([INPUT_IDS], device="cuda")).logits.float().cpu().numpy()
    print("python_forward", {
        "shape": list(py_logits.shape),
        "last_argmax": int(py_logits[0, -1].argmax()),
    })
    del py_model
    gc.collect()
    torch.cuda.empty_cache()

    cpp_model = InferEngine(
        str(model_dir),
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(16, 256),
        attention_backend="paged-attn",
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(cpp_model, str(model_dir), dtype=cpp_model.dtype)
    cpp_logits = raw_cpp_forward(cpp_model, INPUT_IDS, zero_pos)
    diff = np.abs(cpp_logits - py_logits)
    print("cpp_forward", {
        "shape": list(cpp_logits.shape),
        "last_argmax": int(cpp_logits[0, -1].argmax()),
    })
    print("parity", {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "last_argmax_match": bool(cpp_logits[0, -1].argmax() == py_logits[0, -1].argmax()),
    })


if __name__ == "__main__":
    main()
