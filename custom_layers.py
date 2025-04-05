"""
Custom model layers and components for speech recognition models.

This module contains custom neural network layers and components that can be
used to enhance the Whisper model architecture for specific use cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union


class SpecAugment(nn.Module):
    """
    SpecAugment layer for frequency and time masking of spectrograms.
    
    Implementation based on the paper:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
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
            x: Input tensor of shape (batch_size, n_mels, time)
            
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
                freq_start = torch.randint(0, max(1, y.size(1) - freq_mask_width), (1,)).item()
                y[:, freq_start:freq_start + freq_mask_width, :] = 0
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            time_mask_width = torch.randint(0, self.time_mask_param, (1,)).item()
            if time_mask_width > 0:
                time_start = torch.randint(0, max(1, y.size(2) - time_mask_width), (1,)).item()
                y[:, :, time_start:time_start + time_mask_width] = 0
        
        return y


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


class ConvolutionalSubsampler(nn.Module):
    """
    Convolutional subsampling layer for reducing sequence length.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2
    ):
        """
        Initialize convolutional subsampler.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolution
            stride: Stride for convolution
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional subsampling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, in_channels)
            
        Returns:
            Subsampled tensor of shape (batch_size, sequence_length // stride, out_channels)
        """
        # Transpose to (batch_size, in_channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Transpose back to (batch_size, sequence_length, out_channels)
        x = x.transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer models.
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
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


class ResidualConnectionModule(nn.Module):
    """
    Residual connection module with layer normalization.
    """
    
    def __init__(self, module: nn.Module, dropout: float = 0.1):
        """
        Initialize residual connection module.
        
        Args:
            module: Module to wrap with residual connection
            dropout: Dropout probability
        """
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply module with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        return x + self.dropout(self.module(x))


class FeedForwardModule(nn.Module):
    """
    Feed-forward module for transformer models.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward module.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu")
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward module.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CustomAttention(nn.Module):
    """
    Custom multi-head attention module with relative positional encoding.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_relative_pos: bool = False,
        max_relative_position: int = 64
    ):
        """
        Initialize custom attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_relative_pos: Whether to use relative positional encoding
            max_relative_position: Maximum relative position for positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_relative_pos = use_relative_pos
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_relative_pos:
            self.max_relative_position = max_relative_position
            self.rel_pos_embed = nn.Parameter(torch.zeros(2 * max_relative_position + 1, self.head_dim))
            nn.init.xavier_uniform_(self.rel_pos_embed)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply custom attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add relative positional encoding if enabled
        if self.use_relative_pos:
            scores = self._add_relative_position(scores, q, k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def _add_relative_position(
        self,
        scores: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Add relative positional encoding to attention scores.
        
        Args:
            scores: Attention scores
            q: Query tensor
            k: Key tensor
            
        Returns:
            Attention scores with relative positional encoding
        """
        seq_len = scores.size(-1)
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=scores.device)
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
