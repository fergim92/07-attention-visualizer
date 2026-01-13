"""
Transformer Attention Visualizer
==================================
Interactive visualization of attention patterns in transformer models.
Shows how attention heads focus on different parts of the input.
"""

from .models import AttentionOutput
from .transformer import SimplifiedTransformer
from .visualizer import AttentionVisualizer

__version__ = "1.0.0"
__author__ = "Fernando Gimenez"

__all__ = [
    "AttentionOutput",
    "SimplifiedTransformer",
    "AttentionVisualizer",
]
