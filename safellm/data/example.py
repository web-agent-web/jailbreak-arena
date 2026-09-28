"""Example：数据样本抽象，适配自 AISafetyLab ``dataset.example.Example``。

每个样本为一个类字典对象，字段按属性访问。高风险样本与正常样本共用此类。
"""
from __future__ import annotations

import collections.abc


class Example(collections.abc.Mapping):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Example({dict(self)})"
