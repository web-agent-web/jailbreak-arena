"""None 防御：不施加任何防御（基线对照）。"""
from .base import BaseDefender
from .registry import register_defense


@register_defense("none")
class NoneDefense(BaseDefender):
    """直接透传，预处理/过程内/后处理均为 no-op。"""

    def defend(self, env, request):
        messages = request["messages"]
        return {"response": env.query(messages)["response"]}
