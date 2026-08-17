"""Create a one-layer DSV4 checkpoint for an arbitrary layer index from real Mini-1B weights."""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash")
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TARGET = Path(f"/root/autodl-tmp/models/deepseek-v4-mini-1B-layer{LAYER}")

config = json.loads((SOURCE / "config.json").read_text())
config["num_hidden_layers"] = 1
config["compress_ratios"] = [config["compress_ratios"][LAYER]]
config["num_nextn_predict_layers"] = 0
# The renumbered single layer becomes layer 0; force score-based routing so the
# reference does not expect a hash-routing `gate.tid2eid` (only layers < num_hash_layers use it).
config["num_hash_layers"] = 0

prefix = f"layers.{LAYER}."
index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
state = {}
for shard_name in sorted(set(index["weight_map"].values())):
    with safe_open(SOURCE / shard_name, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("mtp."):
                continue
            if name.startswith("layers.") and not name.startswith(prefix):
                continue
            state[name] = handle.get_tensor(name)

# renumber the single layer to index 0 so the model loads it as layer 0
renamed = {}
for name, tensor in state.items():
    new_name = name.replace(prefix, "layers.0.", 1)
    renamed[new_name] = tensor
state = renamed

TARGET.mkdir(parents=True, exist_ok=True)
(TARGET / "config.json").write_text(json.dumps(config, indent=2))
save_file(state, TARGET / "model.safetensors")

for name in ("configuration_deepseek_v4.py", "modeling_deepseek_v4.py"):
    source_file = SOURCE / "code" / "deepseek_v4" / name
    target_file = TARGET / "code" / "deepseek_v4" / name
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(source_file.read_bytes())
(TARGET / "code" / "deepseek_v4" / "__init__.py").write_text("")

print({"target": str(TARGET), "keys": len(state), "compress_ratio": config["compress_ratios"]})
