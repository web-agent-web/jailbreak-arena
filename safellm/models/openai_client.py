"""标准 OpenAI 接口封装。

通过 ``openai`` 官方库访问任意 OpenAI 兼容端点（OpenAI 官方、Azure、vLLM、
本地 LLM Server 等）。满足赛题"大模型访问采用标准的 openai 接口"要求。
"""
from __future__ import annotations

import time
from typing import Any

from ..logging import logger
from .base import Model


class OpenAIChatModel(Model):
    """OpenAI Chat Completions 客户端封装。

    Args:
        model_name: 模型名（如 ``gpt-4o-mini``）。
        base_url: OpenAI 兼容端点。``None`` 时使用 openai 库默认值。
        api_key: API 密钥。``None`` 时使用环境变量 ``OPENAI_API_KEY``。
        generation_config: 默认生成参数（temperature/max_tokens 等）。
        max_retries: 失败最大重试次数。
        retry_gap: 首次重试间隔秒数，每次递增。
        timeout: 单次请求超时秒数。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        generation_config: dict[str, Any] | None = None,
        max_retries: int = 5,
        retry_gap: float = 2.0,
        timeout: float = 60.0,
    ):
        super().__init__(model_name=model_name)
        # 延迟导入，避免无网络/未安装时影响 Mock 模式。
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)
        self.generation_config = generation_config or {}
        self.max_retries = max_retries
        self.retry_gap = retry_gap

    @staticmethod
    def _normalize(messages: list[dict] | str) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def chat(self, messages: list[dict] | str, **kwargs) -> str:
        messages = self._normalize(messages)
        gen_cfg = {**self.generation_config, **kwargs}
        # OpenAI 不接受 generation_config 里的 do_sample 等本地推理参数，过滤。
        for k in ("do_sample", "max_new_tokens"):
            gen_cfg.pop(k, None)

        gap = self.retry_gap
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **gen_cfg,
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("Empty response content from model.")
                self._tick()
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    f"[OpenAIChatModel] 调用失败 (attempt {attempt}/"
                    f"{self.max_retries}): {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(gap)
                    gap += self.retry_gap
        raise RuntimeError(
            f"OpenAI 调用 {self.model_name} 连续 {self.max_retries} 次失败: {last_err}"
        )

    def batch_chat(self, batch_messages: list[list[dict]],
                   max_workers: int = 8, **kwargs) -> list[str]:
        """并发批量推理，保持返回顺序与输入一致。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[str | None] = [None] * len(batch_messages)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {
                ex.submit(self.chat, m, **kwargs): i
                for i, m in enumerate(batch_messages)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"batch_chat[{idx}] 失败: {e}")
                    results[idx] = f"ERROR: {e}"
        return [r if r is not None else "" for r in results]
