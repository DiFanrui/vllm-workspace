import json
import sys
from pathlib import Path


model_dir = Path("/root/autodl-tmp/models/deepseek-v4-mini-1B-from-flash")
config = json.loads((model_dir / "config.json").read_text())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
keys = list(index["weight_map"])

print("config", json.dumps(config, indent=2, ensure_ascii=False))
print("weight_index", {
    "total_size": index.get("metadata", {}).get("total_size"),
    "keys": len(keys),
    "mtp_keys": sum(key.startswith("mtp.") for key in keys),
    "shards": sorted(set(index["weight_map"].values())),
})
print("first_keys", keys[:20])

sys.path.insert(0, str(model_dir / "code"))
from deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

reference_config = DeepseekV4Config.from_pretrained(model_dir)
reference = DeepseekV4ForCausalLM(reference_config)
reference_keys = list(reference.state_dict())
print("python_reference", {
    "parameters": sum(p.numel() for p in reference.parameters()),
    "state_keys": len(reference_keys),
    "first_keys": reference_keys[:10],
})
