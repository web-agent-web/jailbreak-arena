"""模型抽象基类。

沿用 AISafetyLab ``base_model.Model`` 的极简接口：所有模型至少实现 ``chat``。
另提供 ``query_count`` 用于查询预算统计（赛题 §10.3 同分规则 / §11.3 调用次数）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """所有聊天模型的抽象基类。"""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.query_count: int = 0  # 累计成功调用次数

    @abstractmethod
    def chat(self, messages: list[dict] | str, **kwargs) -> str:
        """根据消息列表生成回复，返回纯文本。

        Args:
            messages: OpenAI 风格消息列表 ``[{"role","content"}]``，或纯字符串
                （等价于单条 user 消息）。
        Returns:
            模型回复文本。
        """
        raise NotImplementedError

    def batch_chat(self, batch_messages: list[list[dict]],
                   **kwargs) -> list[str]:
        """顺序批量推理的默认实现；子类可覆盖为并发。"""
        return [self.chat(m, **kwargs) for m in batch_messages]

    def reset_count(self) -> None:
        self.query_count = 0

    def _tick(self) -> None:
        """记录一次成功调用。"""
        self.query_count += 1

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.model_name!r})"
