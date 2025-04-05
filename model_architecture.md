# Speech Recognition Model Architecture

This document provides a detailed overview of the speech recognition model architecture used in this project. The architecture is based on the Whisper model with custom enhancements for improved performance on speech recognition tasks.

## Table of Contents

1. [Overview](#overview)
2. [Base Whisper Architecture](#base-whisper-architecture)
3. [Enhanced Model Components](#enhanced-model-components)
4. [Model Variants](#model-variants)
5. [Implementation Details](#implementation-details)
6. [Configuration Options](#configuration-options)

## Overview

Our speech recognition system is built upon OpenAI's Whisper architecture, which is a Transformer-based encoder-decoder model designed for automatic speech recognition (ASR) and speech translation tasks. We have extended the base architecture with custom components to improve performance, particularly for specific speech recognition scenarios.

The model takes audio input in the form of log-mel spectrograms and outputs text transcriptions. The architecture consists of an encoder that processes the audio input and a decoder that generates the text output.

## Base Whisper Architecture

The base Whisper architecture consists of the following components:

### Encoder

- **Input Processing**: Converts raw audio into log-mel spectrograms with 80 mel bins
- **Convolutional Layers**: A series of 2D convolutional layers to process the spectrogram
- **Positional Encoding**: Adds positional information to the sequence
- **Transformer Encoder Blocks**: Multiple layers of self-attention and feed-forward networks
  - Each block contains:
    - Layer normalization
    - Multi-head self-attention
    - Feed-forward network with GELU activation
    - Residual connections

### Decoder

- **Token Embedding**: Embeds text tokens into a continuous space
- **Positional Encoding**: Adds positional information to the token embeddings
- **Transformer Decoder Blocks**: Multiple layers of self-attention, cross-attention, and feed-forward networks
  - Each block contains:
    - Layer normalization
    - Masked multi-head self-attention
    - Multi-head cross-attention with encoder outputs
    - Feed-forward network with GELU activation
    - Residual connections
- **Output Layer**: Linear projection followed by softmax to produce token probabilities

## Enhanced Model Components

We have enhanced the base Whisper architecture with the following custom components:

### SpecAugment

SpecAugment is a data augmentation technique applied directly within the model. It applies:

- **Time Masking**: Masks random time steps in the spectrogram
- **Frequency Masking**: Masks random frequency bands in the spectrogram

This helps the model become more robust to variations in the input audio and reduces overfitting.

```python
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=27, time_mask_param=100, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def forward(self, x):
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            x = self._freq_mask(x, self.freq_mask_param)
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            x = self._time_mask(x, self.time_mask_param)
        
        return x
```

### Adaptive Layer Normalization

Adaptive Layer Normalization adjusts the normalization parameters based on the input content, allowing for more flexible normalization:

```python
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.adaptor = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, 2)
        )
    
    def forward(self, x):
        # Standard layer normalization
        normalized = self.layer_norm(x)
        
        # Compute adaptive scale and shift
        params = self.adaptor(x.mean(dim=1))
        scale, shift = params[:, 0:1], params[:, 1:2]
        
        # Apply adaptive transformation
        return normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

### Relative Positional Encoding

Instead of using absolute positional encodings, we implement relative positional encodings in the attention mechanism, which helps the model better capture the relative distances between tokens:

```python
class MultiHeadAttentionWithRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_distance + 1, num_heads))
        self.max_distance = max_distance
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        # Implementation details omitted for brevity
        # See models/components.py for full implementation
        pass
```

### Conformer Convolution Module

The Conformer Convolution Module combines convolution and self-attention to capture both local and global dependencies in the audio features:

```python
class ConformerConvModule(nn.Module):
    def __init__(self, embed_dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(
            embed_dim, 2 * embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=kernel_size, stride=1, 
            padding=(kernel_size - 1) // 2, groups=embed_dim
        )
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Implementation details omitted for brevity
        # See models/components.py for full implementation
        pass
```

## Model Variants

We provide two main model variants:

### WhisperModelWrapper

A flexible wrapper around the base Whisper model that adds optional enhancements:

```python
class WhisperModelWrapper(nn.Module):
    def __init__(
        self,
        model_name_or_path="openai/whisper-tiny",
        freeze_encoder=False,
        freeze_decoder=False,
        use_flash_attention=False,
        use_spec_augment=False,
        use_conformer=False,
        use_relative_positional_encoding=False,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        custom_config=None
    ):
        # Implementation details omitted for brevity
        # See models/whisper_model.py for full implementation
        pass
```

### EnhancedWhisperModel

A more advanced implementation that integrates all the custom components for state-of-the-art performance:

```python
class EnhancedWhisperModel(nn.Module):
    def __init__(
        self,
        model_name_or_path="openai/whisper-tiny",
        use_spec_augment=True,
        use_conformer=True,
        use_relative_positional_encoding=True,
        use_adaptive_layer_norm=True,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        custom_config=None
    ):
        # Implementation details omitted for brevity
        # See models/whisper_model.py for full implementation
        pass
```

## Implementation Details

The model implementation is organized into several Python modules:

- `models/whisper_model.py`: Contains the main model classes
- `models/components.py`: Contains the custom components
- `models/config.py`: Contains configuration utilities
- `models/custom_layers.py`: Contains additional custom layers

The implementation uses PyTorch and the Transformers library from Hugging Face, making it compatible with the broader ecosystem of tools and models.

## Configuration Options

The model can be configured with various options to adapt it to different use cases:

### Model Size

- **tiny**: 39M parameters
- **base**: 74M parameters
- **small**: 244M parameters
- **medium**: 769M parameters
- **large**: 1550M parameters

### Language Support

- **English-only**: Models trained specifically for English
- **Multilingual**: Models supporting multiple languages

### Custom Components

- **SpecAugment**: Enable/disable and configure parameters
- **Conformer Modules**: Enable/disable and configure kernel size
- **Relative Positional Encoding**: Enable/disable and configure maximum distance
- **Adaptive Layer Normalization**: Enable/disable

### Training Configuration

- **Freezing Options**: Freeze encoder, decoder, or specific layers
- **Dropout Rates**: Configure dropout for different parts of the model
- **Flash Attention**: Enable/disable for faster training on supported hardware

For detailed configuration options, see the `WhisperModelConfig` class in `models/config.py`.
