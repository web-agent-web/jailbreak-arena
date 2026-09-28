"""runner 子包：全对阵编排、指标计算与报告输出。"""
from .metrics import (
    compute_asr, residual_asr, core_asr, usability_retention,
    aggregate_report,
)
from .arena import Arena

__all__ = [
    "compute_asr", "residual_asr", "core_asr", "usability_retention",
    "aggregate_report", "Arena",
]
