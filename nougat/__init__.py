"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from .model_adapted_extreme import NougatConfig, NougatModel
from .utils.dataset import NougatDataset
from .utils.tpage_dataset import ThreePageDataset
from ._version import __version__

__all__ = [
    "NougatConfig",
    "NougatModel",
    "ThreePageDataset",
]
