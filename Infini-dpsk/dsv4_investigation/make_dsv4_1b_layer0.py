"""Create a one-layer DSV4 checkpoint from the real Mini-1B weights."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash")
VARIANT = os.environ.get("DSV4_LAYER0_VARIANT", "full")
TARGET = Path(f"/root/autodl-tmp/models/deepseek-v4-mini-1B-layer0-{VARIANT}")

config = json.loads((SOURCE / "config.json").read_text())
config["num_hidden_layers"] = 1
config["compress_ratios"] = config["compress_ratios"][:1]
config["num_nextn_predict_layers"] = 0

index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
state = {}
for shard_name in sorted(set(index["weight_map"].values())):
    with safe_open(SOURCE / shard_name, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("mtp."):
                continue
            if (
                name.startswith("layers.")
                and not name.startswith("layers.0.")
            ) or (
                name.startswith("model.layers.")
                and not name.startswith("model.layers.0.")
            ):
                continue
            state[name] = handle.get_tensor(name)

for name, tensor in list(state.items()):
    if not tensor.is_floating_point():
        continue
    if VARIANT == "attention_zero" and name.startswith("layers.0.attn."):
        state[name] = torch.zeros_like(tensor)
    if VARIANT == "ffn_zero" and name.startswith("layers.0.ffn."):
        state[name] = torch.zeros_like(tensor)

TARGET.mkdir(parents=True, exist_ok=True)
(TARGET / "config.json").write_text(json.dumps(config, indent=2))
save_file(state, TARGET / "model.safetensors")

for name in ("configuration_deepseek_v4.py", "modeling_deepseek_v4.py"):
    source_file = SOURCE / "code" / "deepseek_v4" / name
    target_file = TARGET / "code" / "deepseek_v4" / name
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(source_file.read_bytes())
(TARGET / "code" / "deepseek_v4" / "__init__.py").write_text("")

print({"target": str(TARGET), "keys": len(state)})
