# Transformer Attention Visualizer

Interactive visualization of attention patterns in transformer models. Understand how attention heads focus on different parts of the input sequence.

## Overview

This tool provides educational visualizations of the self-attention mechanism, the core component of transformer architectures. It includes a simplified transformer implementation optimized for clarity rather than performance.

## Features

### Visualizations

| Visualization | Description |
|---------------|-------------|
| **Attention Matrix** | Heatmap showing attention weights between all token pairs |
| **Multi-Head View** | Grid of attention patterns across all heads |
| **Attention Flow** | Arrows showing which tokens each position attends to |
| **Positional Encoding** | Sinusoidal patterns used for position information |
| **Q/K/V Projections** | Query, Key, Value representations per head |
| **Attention Entropy** | Focus measure per position (low = focused, high = diffuse) |

### Simplified Transformer

The included transformer implementation features:
- Character-level tokenization
- Sinusoidal positional encodings
- Multi-head self-attention
- Causal (autoregressive) masking option

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## Usage

### Quick Demo

```bash
python attention_visualizer.py
```

This generates:
- `attention_matrix.png` - Single head attention heatmap
- `attention_all_heads.png` - All heads in a grid
- `attention_flow.png` - Attention flow diagram
- `positional_encoding.png` - Position encoding patterns
- `attention_comprehensive.png` - Full analysis dashboard

### Custom Visualization

```python
from attention_visualizer import SimplifiedTransformer, AttentionVisualizer

# Create transformer
transformer = SimplifiedTransformer(
    vocab_size=256,
    d_model=64,
    num_heads=4,
    max_seq_len=128
)

# Process text
text = "Hello world"
attn_output = transformer.forward(text)
tokens = list(text)

# Create visualizer
visualizer = AttentionVisualizer()

# Generate attention matrix
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
visualizer.visualize_attention_matrix(
    attn_output.attention_weights,
    tokens,
    head_idx=0,
    ax=ax
)
plt.savefig('my_attention.png')
```

### Attention Flow

```python
# Visualize which tokens position 5 attends to
fig, ax = plt.subplots(figsize=(14, 6))
visualizer.visualize_attention_flow(
    attn_output.attention_weights,
    tokens,
    query_idx=5,
    top_k=3  # Show top 3 attended positions
)
plt.savefig('attention_flow.png')
```

### Comprehensive Dashboard

```python
fig = visualizer.create_comprehensive_visualization(
    text="The quick brown fox jumps over the lazy dog.",
    transformer=transformer,
    query_idx=4
)
plt.savefig('dashboard.png', dpi=150)
```

## Understanding the Visualizations

### Attention Matrix

```
        Key (attending to)
        t h e   c a t
    t   ■ □ □ □ □ □ □
Q   h   ■ ■ □ □ □ □ □
u   e   ■ ■ ■ □ □ □ □
e       ■ ■ ■ ■ □ □ □
r   c   ■ ■ ■ ■ ■ □ □
y   a   ■ ■ ■ ■ ■ ■ □
    t   ■ ■ ■ ■ ■ ■ ■
```

- **Rows**: Query positions (where attention comes from)
- **Columns**: Key positions (where attention goes to)
- **Color intensity**: Attention weight (darker = higher)
- **Causal mask**: Lower triangular pattern (can't attend to future)

### Attention Entropy

Measures how focused or diffuse attention is:
- **Low entropy**: Attention concentrated on few tokens (focused)
- **High entropy**: Attention spread across many tokens (diffuse)

### Multi-Head Patterns

Different heads often learn different patterns:
- **Local attention**: Focus on nearby tokens
- **Global attention**: Attend to specific positions (e.g., first token)
- **Syntactic attention**: Focus on grammatically related tokens

## Configuration

### Transformer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 256 | Vocabulary size (character-level) |
| `d_model` | 64 | Model/embedding dimension |
| `num_heads` | 4 | Number of attention heads |
| `max_seq_len` | 128 | Maximum sequence length |

### Visualization Options

```python
# Custom colormap for attention
visualizer.attention_cmap = LinearSegmentedColormap.from_list(
    'custom', ['white', 'blue', 'red']
)

# Custom head colors
visualizer.head_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
```

## Output Examples

### Single Head Attention
Shows how one attention head processes the input. Each row shows where that position attends.

### All Heads Grid
Compares attention patterns across heads. Different heads often specialize:
- Head 1: Previous token attention
- Head 2: Beginning of sentence
- Head 3: Similar characters
- Head 4: Word boundaries

### Attention Flow
Visual representation with arrows showing attention direction and strength from a specific query position.

## Technical Details

### Attention Computation

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q = XW_q (Query projection)
- K = XW_k (Key projection)
- V = XW_v (Value projection)
- d_k = head dimension
```

### Multi-Head Attention

```
MultiHead(X) = Concat(head_1, ..., head_h)W_o

Where head_i = Attention(XW_q^i, XW_k^i, XW_v^i)
```

## Extending the Project

### Add Real Model Weights
```python
# Load weights from a trained model
transformer.W_q = loaded_weights['attention.q_proj']
transformer.W_k = loaded_weights['attention.k_proj']
transformer.W_v = loaded_weights['attention.v_proj']
```

### Add BPE Tokenization
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
tokens = tokenizer.encode(text).tokens
```

### Interactive Visualization
```python
import plotly.graph_objects as go
# Create interactive heatmap with hover info
```

## License

MIT License
