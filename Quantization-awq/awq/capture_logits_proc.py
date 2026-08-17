#!/usr/bin/env python3
"""引擎子进程内运行的 LogitsProcessor：把第一次采样的 last-token logits 写到文件。

vLLM 0.21 把 logits_processors 从 SamplingParams 移到了引擎级（LLM(logits_processors=...)）。
该处理器在 EngineCore 子进程里运行，无法直接读主进程的列表 —— 这里用文件跨进程传回：
首次 apply()（即生成第一步的采样 logits）时写入环境变量 AWQ_CAPTURE_PATH 指向的 .npy。

必须在独立模块里定义（顶层、可 import）：config 序列化到 EngineCore 子进程时按引用 pickle。
"""
from __future__ import annotations

import os

import numpy as np

from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class CaptureLogitsProcessor(LogitsProcessor):
    """抓取第一个采样步的整条 last-token logits（[vocab]），写回主进程文件。"""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.captured = False

    def apply(self, logits):
        # 只抓第一次调用：单请求、max_tokens=1 时就是 prompt 最后 token 的 logits。
        if not self.captured:
            self.captured = True
            path = os.environ.get("AWQ_CAPTURE_PATH")
            if path:
                np.save(path, logits[0].float().cpu().numpy())
        return logits

    def is_argmax_invariant(self):
        # 返回 False：进入 non_argmax_invariant 列表，贪心采样也保证被调用。
        return False

    def update_state(self, batch_update):
        pass
