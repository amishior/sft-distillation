# sft_distill/__init__.py

from .config import CONFIG_TEMPLATE
from .pipeline import DatasetDistillationPipeline

__all__ = [
    "CONFIG_TEMPLATE",
    "DatasetDistillationPipeline",
]
