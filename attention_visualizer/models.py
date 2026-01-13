"""
Data models for the Attention Visualizer.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionOutput:
    """Output from attention computation."""
    attention_weights: np.ndarray  # (num_heads, seq_len, seq_len)
    output: np.ndarray  # (seq_len, d_model)
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
