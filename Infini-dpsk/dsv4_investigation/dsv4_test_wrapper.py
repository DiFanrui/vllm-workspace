"""Run InfiniLM's DSV4 precision test with Transformers as its Python reference."""

import runpy
import sys

import transformers.models.deepseek_v4 as deepseek_v4
import transformers.models.deepseek_v4.configuration_deepseek_v4 as configuration
import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling


sys.modules["deepseek_v4"] = deepseek_v4
sys.modules["deepseek_v4.configuration_deepseek_v4"] = configuration
sys.modules["deepseek_v4.modeling_deepseek_v4"] = modeling

runpy.run_path(
    "/root/autodl-tmp/InfiniLM-dpv4-test/test/models/deepseek_v4/"
    "test_deepseek_v4_deterministic_precision.py",
    run_name="__main__",
)
