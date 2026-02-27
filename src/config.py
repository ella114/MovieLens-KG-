from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    positive_threshold: float = 4.0
    top_k: int = 10


DEFAULT_PATHS = Paths()
DEFAULT_TRAIN_CONFIG = TrainConfig()
