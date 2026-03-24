"""Optimized, leakage-safe sales forecasting pipeline."""

from .config import OptimizedTrainingConfig
from .data import load_raw_inputs, prepare_base_frame
from .features import build_feature_frame
from .model import train_optimized_pipeline
