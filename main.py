#!/usr/bin/env python3
"""
Transformer Attention Visualizer - Demo Entry Point
====================================================
Run this script to see the attention visualization in action.

Usage:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from attention_visualizer import SimplifiedTransformer, AttentionVisualizer


def main():
    """Demo of the Attention Visualizer."""
    print("=" * 60)
    print("TRANSFORMER ATTENTION VISUALIZER")
    print("=" * 60)

    # Create transformer
    print("\nInitializing simplified transformer...")
    transformer = SimplifiedTransformer(
        vocab_size=256,
        d_model=64,
        num_heads=4,
        max_seq_len=128
    )

    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Input text: '{text}'")

    # Get attention
    print("\nComputing attention...")
    attn_output = transformer.forward(text)
    tokens = list(text)

    # Create visualizations
    visualizer = AttentionVisualizer()

    # 1. Single head attention matrix
    print("\nGenerating attention matrix visualization...")
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    visualizer.visualize_attention_matrix(
        attn_output.attention_weights,
        tokens,
        head_idx=0,
        ax=ax1,
        title='Attention Head 1'
    )
    plt.savefig('attention_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_matrix.png")

    # 2. All heads
    print("\nGenerating all heads visualization...")
    fig2 = visualizer.visualize_all_heads(attn_output.attention_weights, tokens)
    plt.savefig('attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_all_heads.png")

    # 3. Attention flow
    print("\nGenerating attention flow visualization...")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    visualizer.visualize_attention_flow(
        attn_output.attention_weights,
        tokens,
        query_idx=4,  # "quick"
        ax=ax3
    )
    plt.savefig('attention_flow.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_flow.png")

    # 4. Positional encoding
    print("\nGenerating positional encoding visualization...")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    visualizer.visualize_positional_encoding(
        transformer.position_embedding,
        max_len=50,
        ax=ax4
    )
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Saved: positional_encoding.png")

    # 5. Comprehensive visualization
    print("\nGenerating comprehensive visualization...")
    fig5 = visualizer.create_comprehensive_visualization(
        text,
        transformer,
        query_idx=4
    )
    plt.savefig('attention_comprehensive.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_comprehensive.png")

    # Print statistics
    print("\n" + "=" * 60)
    print("ATTENTION STATISTICS")
    print("=" * 60)

    print(f"\nInput: '{text}'")
    print(f"Sequence length: {len(tokens)}")
    print(f"Number of heads: {transformer.num_heads}")
    print(f"Model dimension: {transformer.d_model}")

    print("\nPer-head attention statistics:")
    for head in range(transformer.num_heads):
        weights = attn_output.attention_weights[head]
        print(f"  Head {head + 1}:")
        print(f"    Max: {weights.max():.4f}")
        print(f"    Mean: {weights.mean():.4f}")
        print(f"    Entropy: {-np.sum(weights * np.log(weights + 1e-10)) / len(tokens):.4f}")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("Generated files:")
    print("  - attention_matrix.png")
    print("  - attention_all_heads.png")
    print("  - attention_flow.png")
    print("  - positional_encoding.png")
    print("  - attention_comprehensive.png")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
