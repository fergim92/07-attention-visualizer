"""
Attention visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional

from .models import AttentionOutput
from .transformer import SimplifiedTransformer


class AttentionVisualizer:
    """
    Visualizer for transformer attention patterns.
    Creates various visualizations of attention mechanisms.
    """

    def __init__(self):
        # Create custom colormaps
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention',
            ['#FFFFFF', '#E3F2FD', '#64B5F6', '#1976D2', '#0D47A1']
        )

        self.head_colors = [
            '#E53935', '#8E24AA', '#1E88E5', '#43A047',
            '#FB8C00', '#00ACC1', '#5E35B1', '#D81B60'
        ]

    def visualize_attention_matrix(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        head_idx: int = 0,
        ax: Optional[plt.Axes] = None,
        title: str = ""
    ) -> plt.Axes:
        """
        Visualize attention weights as a heatmap.

        Args:
            attention_weights: Attention weights (num_heads, seq_len, seq_len)
            tokens: List of token strings
            head_idx: Which attention head to visualize
            ax: Matplotlib axes
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        weights = attention_weights[head_idx]
        seq_len = len(tokens)

        # Plot heatmap
        im = ax.imshow(weights, cmap=self.attention_cmap, aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set ticks
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        # Labels
        ax.set_xlabel('Key (attending to)')
        ax.set_ylabel('Query (from)')
        ax.set_title(title or f'Attention Head {head_idx + 1}')

        return ax

    def visualize_all_heads(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize all attention heads in a grid.

        Args:
            attention_weights: Attention weights (num_heads, seq_len, seq_len)
            tokens: List of token strings
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        num_heads = attention_weights.shape[0]
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        for i in range(num_heads):
            self.visualize_attention_matrix(
                attention_weights,
                tokens,
                head_idx=i,
                ax=axes[i],
                title=f'Head {i + 1}'
            )

        # Hide unused axes
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')

        fig.suptitle('Multi-Head Attention Patterns', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def visualize_attention_flow(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        query_idx: int,
        ax: Optional[plt.Axes] = None,
        top_k: int = 5
    ) -> plt.Axes:
        """
        Visualize attention flow from a specific query position.
        Shows which tokens the query attends to most.

        Args:
            attention_weights: Attention weights
            tokens: Token strings
            query_idx: Index of query token
            ax: Matplotlib axes
            top_k: Number of top attended tokens to highlight

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        num_heads = attention_weights.shape[0]
        seq_len = len(tokens)

        # Average attention across heads
        avg_attention = np.mean(attention_weights[:, query_idx, :], axis=0)

        # Create visualization
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(-1, num_heads + 1)

        # Draw tokens at bottom
        for i, token in enumerate(tokens):
            # Background based on attention
            color = self.attention_cmap(avg_attention[i])
            rect = Rectangle((i - 0.4, -0.8), 0.8, 0.6,
                            facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(i, -0.5, token, ha='center', va='center', fontsize=9)

        # Draw attention from each head
        for head in range(num_heads):
            # Query token position
            ax.scatter(query_idx, head + 0.5, s=200, c=self.head_colors[head % len(self.head_colors)],
                      marker='s', zorder=5, label=f'Head {head + 1}' if head == 0 else None)
            ax.text(query_idx, head + 0.5, tokens[query_idx], ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')

            # Draw arrows to attended tokens
            weights = attention_weights[head, query_idx, :]
            top_indices = np.argsort(weights)[-top_k:]

            for key_idx in top_indices:
                if key_idx != query_idx:
                    weight = weights[key_idx]
                    arrow = FancyArrowPatch(
                        (query_idx, head + 0.3),
                        (key_idx, 0),
                        arrowstyle='-|>',
                        mutation_scale=10,
                        color=self.head_colors[head % len(self.head_colors)],
                        alpha=weight,
                        linewidth=weight * 3
                    )
                    ax.add_patch(arrow)

        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([f'Head {i+1}' for i in range(num_heads)])
        ax.set_xlabel('Token Position')
        ax.set_title(f'Attention Flow from "{tokens[query_idx]}" (position {query_idx})')
        ax.grid(True, alpha=0.3)

        return ax

    def visualize_positional_encoding(
        self,
        encoding: np.ndarray,
        max_len: int = 50,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Visualize positional encoding patterns.

        Args:
            encoding: Positional encoding matrix
            max_len: Maximum length to display
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        pe = encoding[:max_len, :64]  # Show first 64 dimensions

        im = ax.imshow(pe.T, cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Embedding Dimension')
        ax.set_title('Positional Encoding Patterns')
        plt.colorbar(im, ax=ax)

        return ax

    def visualize_qkv_projections(
        self,
        attention_output: AttentionOutput,
        tokens: List[str],
        head_idx: int = 0,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Visualize Query, Key, Value projections.

        Args:
            attention_output: Output from attention computation
            tokens: Token strings
            head_idx: Which head to visualize
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        projections = [
            ('Query', attention_output.query[head_idx]),
            ('Key', attention_output.key[head_idx]),
            ('Value', attention_output.value[head_idx])
        ]

        for ax, (name, proj) in zip(axes, projections):
            im = ax.imshow(proj, cmap='RdBu', aspect='auto')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Position')
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_title(f'{name} Projection (Head {head_idx + 1})')
            plt.colorbar(im, ax=ax)

        fig.suptitle('Query, Key, Value Projections', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def create_comprehensive_visualization(
        self,
        text: str,
        transformer: SimplifiedTransformer,
        query_idx: int = 0,
        figsize: Tuple[int, int] = (18, 14)
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of the attention mechanism.

        Args:
            text: Input text
            transformer: Transformer model
            query_idx: Query position for flow visualization
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Get attention output
        attn_output = transformer.forward(text)
        tokens = list(text)[:20]  # Limit for visibility

        # Create figure with gridspec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Attention matrix (head 0)
        ax1 = fig.add_subplot(gs[0, 0])
        self.visualize_attention_matrix(
            attn_output.attention_weights[:, :len(tokens), :len(tokens)],
            tokens, head_idx=0, ax=ax1, title='Head 1 Attention'
        )

        # 2. Attention matrix (head 1)
        ax2 = fig.add_subplot(gs[0, 1])
        if transformer.num_heads > 1:
            self.visualize_attention_matrix(
                attn_output.attention_weights[:, :len(tokens), :len(tokens)],
                tokens, head_idx=1, ax=ax2, title='Head 2 Attention'
            )
        else:
            ax2.axis('off')

        # 3. Average attention
        ax3 = fig.add_subplot(gs[0, 2])
        avg_attn = np.mean(attn_output.attention_weights, axis=0)[:len(tokens), :len(tokens)]
        im = ax3.imshow(avg_attn, cmap=self.attention_cmap)
        ax3.set_xticks(range(len(tokens)))
        ax3.set_yticks(range(len(tokens)))
        ax3.set_xticklabels(tokens, rotation=45, ha='right')
        ax3.set_yticklabels(tokens)
        ax3.set_title('Average Attention (All Heads)')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # 4. Attention flow
        ax4 = fig.add_subplot(gs[1, :])
        self.visualize_attention_flow(
            attn_output.attention_weights[:, :len(tokens), :len(tokens)],
            tokens, query_idx=min(query_idx, len(tokens)-1), ax=ax4
        )

        # 5. Positional encoding
        ax5 = fig.add_subplot(gs[2, 0])
        self.visualize_positional_encoding(transformer.position_embedding, max_len=30, ax=ax5)

        # 6. Attention entropy (measure of focus)
        ax6 = fig.add_subplot(gs[2, 1])
        entropies = []
        for head in range(transformer.num_heads):
            attn = attn_output.attention_weights[head, :len(tokens), :len(tokens)]
            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1)
            entropies.append(entropy)
        entropies = np.array(entropies)

        for head in range(transformer.num_heads):
            ax6.plot(entropies[head], label=f'Head {head+1}',
                    color=self.head_colors[head % len(self.head_colors)])
        ax6.set_xlabel('Position')
        ax6.set_ylabel('Entropy')
        ax6.set_title('Attention Entropy (Focus Measure)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Attention statistics
        ax7 = fig.add_subplot(gs[2, 2])
        stats_text = [
            f"Input length: {len(tokens)} tokens",
            f"Number of heads: {transformer.num_heads}",
            f"Model dimension: {transformer.d_model}",
            f"Head dimension: {transformer.head_dim}",
            "",
            "Attention Statistics:",
            f"  Max attention: {attn_output.attention_weights.max():.3f}",
            f"  Min attention: {attn_output.attention_weights.min():.3f}",
            f"  Mean attention: {attn_output.attention_weights.mean():.3f}",
        ]
        ax7.text(0.1, 0.9, '\n'.join(stats_text), transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax7.axis('off')
        ax7.set_title('Model Statistics')

        fig.suptitle(f'Transformer Attention Analysis: "{text[:30]}..."',
                    fontsize=14, fontweight='bold')

        return fig
