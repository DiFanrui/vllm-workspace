"""Run the InfiniLM DeepSeek-V4 tiny checkpoint through its C++ CUDA engine."""

import importlib.util
import ctypes
from pathlib import Path

import numpy as np


TEST = Path(
    "/root/autodl-tmp/InfiniLM-dpv4-test/test/models/deepseek_v4/"
    "test_deepseek_v4_deterministic_precision.py"
)
spec = importlib.util.spec_from_file_location("dsv4_precision", TEST)
precision = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(precision)

model_dir = precision.write_tiny_model(Path("/root/autodl-tmp/dpv4-tiny-cpp"))
model = precision.load_cpp_model(model_dir, "cuda")
input_ids = [3, 5, 7, 11, 13]
raw = super(precision.InferEngine, model)
seq = len(input_ids)
block = model.get_cache_config().block_size()
max_blocks = (seq + 2 + block - 1) // block


def underlying(tensor):
    return tensor._underlying if hasattr(tensor, "_underlying") else tensor


output = raw.forward(
    raw.Input(
        underlying(precision.infinicore.from_list([input_ids], dtype=precision.infinicore.int64).view([1, seq])),
        position_ids=underlying(precision.infinicore.from_list(list(range(seq)), dtype=precision.infinicore.int64)),
        past_sequence_lengths=underlying(precision.infinicore.from_list([0], dtype=precision.infinicore.int32)),
        total_sequence_lengths=underlying(precision.infinicore.from_list([seq], dtype=precision.infinicore.int32)),
        input_offsets=underlying(precision.infinicore.from_list([0, seq], dtype=precision.infinicore.int32)),
        cu_seqlens=underlying(precision.infinicore.from_list([0, seq], dtype=precision.infinicore.int32)),
        block_tables=underlying(
            precision.infinicore.from_list([list(range(max_blocks))], dtype=precision.infinicore.int32)
        ),
        slot_mapping=underlying(precision.infinicore.from_list(list(range(seq)), dtype=precision.infinicore.int64)),
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )
)
logits_bf16 = precision.infinicore.Tensor(output.logits).to(
    precision.infinicore.device("cpu", 0)
)
raw_bits = np.ctypeslib.as_array(
    (ctypes.c_uint16 * logits_bf16.numel()).from_address(logits_bf16.data_ptr())
).copy()
logits = (raw_bits.astype(np.uint32) << 16).view(np.float32).reshape(logits_bf16.shape)
print(
    "cpp_forward:",
    {
        "shape": list(logits.shape),
        "finite": bool(np.isfinite(logits).all()),
        "last_argmax": int(logits.reshape(-1, logits.shape[-1])[-1].argmax()),
        "max_abs": float(np.abs(logits).max()),
    },
)

out = model.generate(
    precision.infinicore.from_list([input_ids], dtype=precision.infinicore.int64),
    precision.GenerationConfig(
        max_new_tokens=2,
        eos_token_id=[],
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        stop_on_eos=False,
    ),
)
tokens = []
for tensor in out if isinstance(out, list) else [out]:
    tokens.extend(int(x) for x in tensor.to_numpy().reshape(-1).tolist())
print("cpp_generate:", tokens)
