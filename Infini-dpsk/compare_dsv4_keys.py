import json
from pathlib import Path

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import _remap_deepseek_v4, _is_internal_moe_packed_weight


model_dir = Path("/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash")
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
checkpoint_keys = set(_remap_deepseek_v4({key: None for key in index["weight_map"]}))

model = InferEngine(
    str(model_dir),
    device=infinicore.device("cuda", 0),
    distributed_config=DistConfig(1),
    cache_config=PagedKVCacheConfig(16, 64),
    attention_backend="paged-attn",
    weight_load_mode="sync",
)
model_keys = set(model.state_dict_keyname())
missing = sorted(
    key for key in model_keys - checkpoint_keys if not _is_internal_moe_packed_weight(key)
)
unexpected = sorted(checkpoint_keys - model_keys)
print({
    "model_keys": len(model_keys),
    "checkpoint_keys_after_remap": len(checkpoint_keys),
    "missing": len(missing),
    "unexpected": len(unexpected),
})
print("missing_keys", missing)
print("unexpected_keys", unexpected)
