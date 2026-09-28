"""攻防解耦环境层。

赛题定义两个互不可见的接口：
- ``DefenseEnv``（赛题 §6.2）：防御方用它调用 *原始* 目标模型。``query`` 直连
  目标模型，系统提示词由裁判固定，防御方不得删除/替换/覆盖。
- ``AttackEnv``（赛题 §5.3）：攻击方用它调用 *经当前防御保护* 的目标模型。
  ``AttackEnv.query`` 内部构造一个 ``DefenseEnv`` 并调用当前防御方的 ``defend``。

由此实现解耦：攻击方只见 ``AttackEnv``，防御方只见 ``DefenseEnv``，双方互不
感知对方身份/实现/状态。攻击方法与防御算法均通过注册表装配，arena 按配置注入。
"""
from .defense_env import DefenseEnv
from .attack_env import AttackEnv

__all__ = ["DefenseEnv", "AttackEnv"]
