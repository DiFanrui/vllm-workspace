import ctypes
import gc
import json
import os
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


MODEL_DIR = Path(
    os.environ.get(
        "DSV4_MODEL_DIR",
        "/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash",
    )
)
INPUT_IDS = [3, 5, 7, 11, 13]


def bf16_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    cpu = tensor.to(infinicore.device("cpu", 0))
    bits = np.ctypeslib.as_array(
        (ctypes.c_uint16 * cpu.numel()).from_address(cpu.data_ptr())
    ).copy()
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(cpu.shape)


def raw_cpp_forward(model: InferEngine, input_ids: list[int]) -> np.ndarray:
    raw = super(InferEngine, model)
    seq = len(input_ids)
    block = model.get_cache_config().block_size()
    max_blocks = (seq + 2 + block - 1) // block

    def u(tensor):
        return tensor._underlying if hasattr(tensor, "_underlying") else tensor

    positions = (
        [0] * seq
        if os.environ.get("DSV4_ZERO_POSITIONS") == "1"
        else list(range(seq))
    )
    output = raw.forward(
        raw.Input(
            u(infinicore.from_list([input_ids], dtype=infinicore.int64).view([1, seq])),
            position_ids=u(infinicore.from_list(positions, dtype=infinicore.int64)),
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
    index_path = MODEL_DIR / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        expected_keys = set(index["weight_map"])
        shard_names = set(index["weight_map"].values())
        tensor_bytes = int(index["metadata"]["total_size"])
    else:
        shard_names = {path.name for path in MODEL_DIR.glob("*.safetensors")}
        expected_keys = set()
        tensor_bytes = 0
        for name in shard_names:
            with safe_open(MODEL_DIR / name, framework="pt", device="cpu") as handle:
                expected_keys.update(handle.keys())
                tensor_bytes += sum(
                    handle.get_tensor(key).numel()
                    * handle.get_tensor(key).element_size()
                    for key in handle.keys()
                )
    actual_keys = set()
    file_bytes = 0
    for name in shard_names:
        shard = MODEL_DIR / name
        file_bytes += shard.stat().st_size
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual_keys.update(handle.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"checkpoint key mismatch: missing={expected_keys - actual_keys}, "
            f"unexpected={actual_keys - expected_keys}"
        )
    print("checkpoint_integrity: PASS", {
        "tensor_bytes": tensor_bytes,
        "file_bytes": file_bytes,
        "keys": len(actual_keys),
    })

    sys.path.insert(0, str(MODEL_DIR / "code"))
    from deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    t0 = time.perf_counter()
    py_config = DeepseekV4Config.from_pretrained(MODEL_DIR)
    py_model = DeepseekV4ForCausalLM.from_pretrained(
        MODEL_DIR, config=py_config, torch_dtype=torch.bfloat16
    ).cuda().eval()
    if os.environ.get("DSV4_PY_QUANTIZE_HC_FN") == "1":
        quantized = 0
        with torch.no_grad():
            for name, parameter in py_model.named_parameters():
                if name.endswith(("_fn", "_fn.weight")) or "hc_" in name and "_fn" in name:
                    parameter.copy_(parameter.bfloat16().float())
                    quantized += 1
        print("python_quantized_hc_fn", quantized)
    with torch.inference_mode():
        py_kwargs = {}
        if os.environ.get("DSV4_ZERO_POSITIONS") == "1":
            py_kwargs["position_ids"] = torch.zeros(
                (1, len(INPUT_IDS)), dtype=torch.long, device="cuda"
            )
        py_logits = py_model(
            torch.tensor([INPUT_IDS], device="cuda"), **py_kwargs
        ).logits.float().cpu().numpy()
    print("python_forward", {
        "seconds": round(time.perf_counter() - t0, 3),
        "shape": list(py_logits.shape),
        "finite": bool(np.isfinite(py_logits).all()),
        "last_argmax": int(py_logits[0, -1].argmax()),
    })
    del py_model
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    cpp_model = InferEngine(
        str(MODEL_DIR),
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(16, 256),
        attention_backend="paged-attn",
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(cpp_model, str(MODEL_DIR), dtype=cpp_model.dtype)
    load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    cpp_logits = raw_cpp_forward(cpp_model, INPUT_IDS)
    forward_seconds = time.perf_counter() - t0
    diff = np.abs(cpp_logits - py_logits)
    print("cpp_forward", {
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "shape": list(cpp_logits.shape),
        "finite": bool(np.isfinite(cpp_logits).all()),
        "last_argmax": int(cpp_logits[0, -1].argmax()),
    })
    print("parity", {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "last_argmax_match": bool(cpp_logits[0, -1].argmax() == py_logits[0, -1].argmax()),
    })

    generated = cpp_model.generate(
        infinicore.from_list([INPUT_IDS], dtype=infinicore.int64),
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=[],
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            stop_on_eos=False,
        ),
    )
    tokens = []
    for tensor in generated if isinstance(generated, list) else [generated]:
        tokens.extend(int(x) for x in tensor.to_numpy().reshape(-1).tolist())
    print("cpp_generate", tokens)


if __name__ == "__main__":
    main()
