"""
Simplified Transformer implementation for visualization.
"""

import numpy as np
from typing import List, Optional

from .models import AttentionOutput


class SimplifiedTransformer:
    """
    A simplified transformer implementation for attention visualization.
    Not optimized for performance, but for educational clarity.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        num_heads: int = 4,
        max_seq_len: int = 128
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Initialize embeddings
        np.random.seed(42)
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = self._create_positional_encoding()

        # Initialize attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pe = np.zeros((self.max_seq_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def tokenize(self, text: str) -> List[int]:
        """Simple character-level tokenization."""
        return [ord(c) % self.vocab_size for c in text]

    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Get token + positional embeddings."""
        seq_len = len(token_ids)
        token_emb = self.token_embedding[token_ids]
        pos_emb = self.position_embedding[:seq_len]
        return token_emb + pos_emb

    def multi_head_attention(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> AttentionOutput:
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            AttentionOutput with attention weights and output
        """
        seq_len = x.shape[0]

        # Linear projections
        Q = x @ self.W_q  # (seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        # (seq_len, d_model) -> (num_heads, seq_len, head_dim)
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Compute attention scores
        # (num_heads, seq_len, head_dim) @ (num_heads, head_dim, seq_len) -> (num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)

        # Apply mask if provided (causal attention)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax
        attention_weights = self._softmax(scores)

        # Apply attention to values
        # (num_heads, seq_len, seq_len) @ (num_heads, seq_len, head_dim) -> (num_heads, seq_len, head_dim)
        context = attention_weights @ V

        # Reshape back
        # (num_heads, seq_len, head_dim) -> (seq_len, d_model)
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # Output projection
        output = context @ self.W_o

        return AttentionOutput(
            attention_weights=attention_weights,
            output=output,
            query=Q,
            key=K,
            value=V
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, text: str, use_causal_mask: bool = True) -> AttentionOutput:
        """
        Forward pass through the transformer.

        Args:
            text: Input text
            use_causal_mask: Whether to use causal (autoregressive) masking

        Returns:
            AttentionOutput with all intermediate values
        """
        # Tokenize
        token_ids = self.tokenize(text)
        seq_len = len(token_ids)

        # Get embeddings
        x = self.get_embeddings(token_ids)

        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = np.tril(np.ones((seq_len, seq_len)))

        # Compute attention
        return self.multi_head_attention(x, mask)
