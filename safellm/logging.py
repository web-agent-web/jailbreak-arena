"""日志兼容层。

优先使用 loguru（与 AISafetyLab 一致的接口），缺失时回退到标准库 logging，
保证后端在无 loguru 环境下亦可运行。对外暴露统一的 ``logger`` 对象，提供
``info/debug/warning/error`` 方法。
"""
from __future__ import annotations

try:  # pragma: no cover - 依赖是否存在取决于运行环境
    from loguru import logger  # type: ignore

    _HAS_LOGURU = True
except Exception:  # pragma: no cover
    import logging
    import sys

    _logger = logging.getLogger("safellm")
    if not _logger.handlers:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        _logger.addHandler(_handler)
        _logger.setLevel(logging.INFO)

    class _LoguruShim:
        """loguru 子集接口的 stdlog 实现，仅满足本项目使用。"""

        def info(self, msg, *args, **kwargs):
            _logger.info(msg, *args, **kwargs)

        def debug(self, msg, *args, **kwargs):
            _logger.debug(msg, *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            _logger.warning(msg, *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            _logger.error(msg, *args, **kwargs)

        def success(self, msg, *args, **kwargs):
            _logger.info(msg, *args, **kwargs)

    logger = _LoguruShim()  # type: ignore
    _HAS_LOGURU = False

__all__ = ["logger"]
