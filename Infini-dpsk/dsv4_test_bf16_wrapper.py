"""Add BF16 NumPy conversion missing from InfiniLM's DSV4 precision test."""

import ctypes
import runpy

import numpy as np

import infinicore
import infinilm.generation.utils  # installs the original Tensor.to_numpy


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
runpy.run_path(
    "/root/autodl-tmp/InfiniLM-dpv4-test/test/models/deepseek_v4/"
    "test_deepseek_v4_deterministic_precision.py",
    run_name="__main__",
)
