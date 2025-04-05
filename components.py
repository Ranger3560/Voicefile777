"""
Advanced model components for speech recognition models.

This module contains advanced model components that can be used to enhance
the performance of Whisper models for speech recognition tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and loaded)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 64, dropout: float = 0.1):
        """
        Initialize relative positional encoding.
        
        Args:
            d_model: Model dimension
            max_relative_position: Maximum relative position
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        
        # Create relative positional encoding
        self.relative_position_embedding = nn.Parameter(
            torch.zeros(2 * max_relative_position + 1, d_model)
        )
        nn.init.xavier_uniform_(self.relative_position_embedding)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input tensor and return relative positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Tuple of (processed input, relative positional encoding)
        """
        seq_length = x.size(1)
        
        # Create relative position indices
        range_vec = torch.arange(seq_length, device=x.device)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        
        # Clip to max_relative_position
        range_mat = torch.clamp(range_mat, -self.max_relative_position, self.max_relative_position)
        
        # Shift indices to be non-negative
        range_mat = range_mat + self.max_relative_position
        
        # Get relative positional embeddings
        relative_position_encoding = self.relative_position_embedding[range_mat]
        
        return self.dropout(x), relative_position_encoding


class ConvSubsampler(nn.Module):
    """
    Convolutional subsampling layer for reducing sequence length.
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 3],
        strides: List[int] = [2, 2]
    ):
        """
        Initialize convolutional subsampler.
        
        Args:
            in_channels: Number of input channels
            mid_channels: Number of middle channels
            out_channels: Number of output channels
            kernel_sizes: List of kernel sizes for each conv layer
            strides: List of strides for each conv layer
        """
        super().__init__()
        assert len(kernel_sizes) == len(strides), "Must provide same number of kernel sizes and strides"
        
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels,
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    padding=(kernel_sizes[0] - 1) // 2
                ),
                nn.ReLU()
            )
        )
        
        # Remaining conv layers
        for i in range(1, len(kernel_sizes)):
            channels = mid_channels if i < len(kernel_sizes) - 1 else out_channels
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        mid_channels, channels,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=(kernel_sizes[i] - 1) // 2
                    ),
                    nn.ReLU()
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional subsampling.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time, freq)
            
        Returns:
            Subsampled tensor
        """
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape to (batch_size, time, freq * channels)
        batch_size, channels, time, freq = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, time, channels * freq)
        
        return x


class SpecAugment(nn.Module):
    """
    SpecAugment layer for frequency and time masking of spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 10,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Initialize SpecAugment layer.
        
        Args:
            freq_mask_param: Maximum width of frequency masks
            time_mask_param: Maximum width of time masks
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time, freq)
            
        Returns:
            Augmented tensor
        """
        if not self.training:
            return x
        
        y = x.clone()
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            freq_mask_width = torch.randint(0, self.freq_mask_param, (1,)).item()
            if freq_mask_width > 0:
                freq_start = torch.randint(0, max(1, y.size(3) - freq_mask_width), (1,)).item()
                y[:, :, :, freq_start:freq_start + freq_mask_width] = 0
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            time_mask_width = torch.randint(0, self.time_mask_param, (1,)).item()
            if time_mask_width > 0:
                time_start = torch.randint(0, max(1, y.size(2) - time_mask_width), (1,)).item()
                y[:, :, time_start:time_start + time_mask_width, :] = 0
        
        return y


class MultiHeadAttentionWithRelPos(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        max_relative_position: int = 64
    ):
        """
        Initialize multi-head attention with relative positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            add_bias_kv: Whether to add bias to key and value projections
            add_zero_attn: Whether to add zero attention
            max_relative_position: Maximum relative position for positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = max_relative_position
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Relative positional encoding
        self.rel_pos_embed = nn.Parameter(torch.zeros(2 * max_relative_position + 1, self.head_dim))
        nn.init.xavier_uniform_(self.rel_pos_embed)
        
        # Additional parameters
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention with relative positional encoding.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Key padding mask
            attn_mask: Attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Add bias to key and value if specified
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(batch_size, 1, 1, 1)], dim=2)
            v = torch.cat([v, self.bias_v.repeat(batch_size, 1, 1, 1)], dim=2)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(batch_size, 1)], dim=1
                )
            src_len += 1
        
        # Add zero attention if specified
        if self.add_zero_attn:
            k = torch.cat([k, torch.zeros_like(k[:, :, :1])], dim=2)
            v = torch.cat([v, torch.zeros_like(v[:, :, :1])], dim=2)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(batch_size, 1)], dim=1
                )
            src_len += 1
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional encoding
        scores = self._add_relative_position(scores, q)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def _add_relative_position(self, scores: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Add relative positional encoding to attention scores.
        
        Args:
            scores: Attention scores
            q: Query tensor
            
        Returns:
            Attention scores with relative positional encoding
        """
        tgt_len, src_len = scores.size(-2), scores.size(-1)
        
        # Create relative position indices
        range_vec = torch.arange(src_len, device=scores.device)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        
        # Clip to max_relative_position
        range_mat = torch.clamp(range_mat, -self.max_relative_position, self.max_relative_position)
        
        # Shift indices to be non-negative
        range_mat = range_mat + self.max_relative_position
        
        # Get relative positional embeddings
        rel_pos = self.rel_pos_embed[range_mat]
        
        # Compute relative positional attention
        rel_scores = torch.matmul(q.unsqueeze(-2), rel_pos.transpose(-2, -1)).squeeze(-2)
        
        # Add to original scores
        return scores + rel_scores


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with learnable gain and bias parameters.
    """
    
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size]):
        """
        Initialize Adaptive Layer Normalization.
        
        Args:
            normalized_shape: Shape of the input tensor
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        normalized = self.layer_norm(x)
        return normalized * self.gain + self.bias


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network with configurable activation function.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "swish")
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

(Content truncated due to size limit. Use line ranges to read in chunks)