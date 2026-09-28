"""
SafeLLM Jailbreak Arena 后端核心包。

本包基于 AISafetyLab 的算法与架构模式实现赛题《提示词越狱攻防赛题设计.md》
所定义的攻防对抗后端。攻击方法与防御算法通过注册表解耦，LLM 访问统一走
标准 OpenAI 接口。
"""

__version__ = "0.1.0"
