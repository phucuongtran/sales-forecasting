from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class OptimizedTrainingConfig:
    data_dir: str = "data"
    models_dir: str = "models"
    cutoff_date: str = "2017-10-01"
    n_optuna_trials: int = 40
    n_splits: int = 4
    validation_days: int = 28
    gap_days: int = 7
    min_train_days: int = 180
    ensemble_size: int = 3
    random_seed: int = 2025
    use_log_target: bool = True
    objective_options: List[str] = field(
        default_factory=lambda: ["regression", "poisson", "tweedie"]
    )
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    ewm_alphas: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])
    global_windows: List[int] = field(default_factory=lambda: [7, 28])
    low_sales_floor: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)
