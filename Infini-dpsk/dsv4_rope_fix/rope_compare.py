"""Direct comparison of C++ RoPE (GPT_NEOX / GPT_J) vs reference split-half RoPE.

Full rotation case (rotary_dim == head_dim == D) on 4D tensor [B, S, H, D].
"""

import ctypes

import numpy as np
import torch

import infinicore
from infinicore.nn.modules.rope import RoPE
from infinicore.nn.functional.rope import RopeAlgo


def ref_split_half(x, positions, base, dim):
    """Reference build_rope_cache + _rotate_half (split-half / GPT-NeoX)."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))  # [half]
    out = x.float().clone()
    for si, p in enumerate(positions.tolist()):
        ang = p * inv_freq
        c = torch.cos(ang).to(torch.float32)
        s = torch.sin(ang).to(torch.float32)
        for i in range(half):
            x0 = out[0, si, 0, i]
            x1 = out[0, si, 0, i + half]
            out[0, si, 0, i] = x0 * c[i] - x1 * s[i]
            out[0, si, 0, i + half] = x0 * s[i] + x1 * c[i]
    return out


def bf16_to_np(t):
    cpu = t.to(infinicore.device("cpu", 0))
    bits = np.ctypeslib.as_array((ctypes.c_uint16 * cpu.numel()).from_address(cpu.data_ptr())).copy()
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(cpu.shape)


def main():
    D = 32
    S = 5
    base = 10000.0
    torch.manual_seed(0)
    x = torch.randn(1, S, 1, D, dtype=torch.bfloat16)
    positions = torch.arange(S, dtype=torch.long)

    ref = ref_split_half(x, positions, base, D)

    x_inf = infinicore.from_numpy(x.float().numpy(), dtype=infinicore.bfloat16,
                                  device=infinicore.device("cpu", 0))
    pos_inf = infinicore.from_numpy(positions.numpy(), dtype=infinicore.int64)

    for algo in (RopeAlgo.GPT_NEOX, RopeAlgo.GPT_J):
        rope = RoPE(max_position_embeddings=S, rope_theta=base, head_dim=D, rotary_dim=D,
                    device=infinicore.device("cpu", 0), dtype=infinicore.bfloat16)
        out = rope.forward(x_inf, pos_inf, algo=algo)
        out_np = bf16_to_np(out)
        diff = np.abs(out_np - ref.float().numpy())
        print(f"algo={algo}: max_abs={diff.max():.6f} mean_abs={diff.mean():.6f}")
        print("  cpp row0 =", np.round(out_np[0, 0, 0, :8], 4).tolist())
    print("  ref row0 =", np.round(ref.float().numpy()[0, 0, 0, :8], 4).tolist())


if __name__ == "__main__":
    main()
