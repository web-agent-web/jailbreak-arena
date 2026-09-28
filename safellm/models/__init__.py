"""模型层：统一 ``chat`` 接口，支持标准 OpenAI 兼容端点与离线 Mock。"""
from .base import Model
from .openai_client import OpenAIChatModel
from .mock_model import MockChatModel, create_model

__all__ = ["Model", "OpenAIChatModel", "MockChatModel", "create_model"]
