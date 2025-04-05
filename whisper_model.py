"""
Whisper model implementation and configuration for speech recognition.

This module contains the implementation of the Whisper model architecture
and configuration utilities for fine-tuning on custom datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
import logging
import os
import sys

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components import (
    SpecAugment,
    AdaptiveLayerNorm,
    MultiHeadAttentionWithRelPos,
    ConformerConvModule,
    ResidualConnectionModule
)
from models.config import WhisperModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperModelWrapper(nn.Module):
    """
    Wrapper class for Whisper model with additional functionality for fine-tuning.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-tiny",
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        use_flash_attention: bool = False,
        use_spec_augment: bool = False,
        use_conformer: bool = False,
        use_relative_positional_encoding: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        custom_config: Optional[WhisperModelConfig] = None
    ):
        """
        Initialize the Whisper model wrapper.
        
        Args:
            model_name_or_path: Name or path of the pre-trained Whisper model
            freeze_encoder: Whether to freeze the encoder parameters
            freeze_decoder: Whether to freeze the decoder parameters
            use_flash_attention: Whether to use flash attention for faster training
            use_spec_augment: Whether to use SpecAugment for data augmentation
            use_conformer: Whether to use Conformer convolution modules
            use_relative_positional_encoding: Whether to use relative positional encoding
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            activation_dropout: Activation dropout probability
            custom_config: Custom model configuration
        """
        super().__init__()
        
        # Load pre-trained model
        if custom_config is not None:
            logger.info(f"Creating model from custom configuration")
            self.model = custom_config.create_model()
            
            # Load pretrained weights if available
            try:
                pretrained_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
                
                # Copy compatible weights
                pretrained_state_dict = pretrained_model.state_dict()
                model_state_dict = self.model.state_dict()
                
                for name, param in pretrained_state_dict.items():
                    if name in model_state_dict and param.shape == model_state_dict[name].shape:
                        model_state_dict[name].copy_(param)
                
                logger.info(f"Loaded compatible pretrained weights from {model_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
        else:
            logger.info(f"Loading pretrained model from {model_name_or_path}")
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        
        # Get model configuration
        self.config = self.model.config
        
        # Update dropout if specified
        if dropout != 0.1 or attention_dropout > 0.0 or activation_dropout > 0.0:
            self._update_dropout(dropout, attention_dropout, activation_dropout)
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing encoder parameters")
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        
        # Freeze decoder if specified
        if freeze_decoder:
            logger.info("Freezing decoder parameters")
            for param in self.model.model.decoder.parameters():
                param.requires_grad = False
        
        # Enable flash attention if specified and available
        if use_flash_attention:
            self._enable_flash_attention()
        
        # Add SpecAugment if specified
        self.use_spec_augment = use_spec_augment
        if use_spec_augment:
            logger.info("Adding SpecAugment layer")
            self.spec_augment = SpecAugment(
                freq_mask_param=27,  # ~10% of mel bins
                time_mask_param=100,  # ~10% of max sequence length
                n_freq_masks=2,
                n_time_masks=2
            )
        
        # Add Conformer modules if specified
        self.use_conformer = use_conformer
        if use_conformer:
            logger.info("Adding Conformer convolution modules")
            self.conformer_layers = nn.ModuleList([
                ResidualConnectionModule(
                    ConformerConvModule(
                        embed_dim=self.config.d_model,
                        kernel_size=31,
                        dropout=dropout
                    ),
                    dropout=dropout
                )
                for _ in range(min(4, self.config.encoder_layers))
            ])
        
        # Add relative positional encoding if specified
        self.use_relative_positional_encoding = use_relative_positional_encoding
        if use_relative_positional_encoding:
            logger.info("Adding relative positional encoding")
            self._replace_attention_with_relative_pos()
    
    def _update_dropout(
        self,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float
    ):
        """
        Update dropout probability in the model.
        
        Args:
            dropout: New dropout probability
            attention_dropout: New attention dropout probability
            activation_dropout: New activation dropout probability
        """
        logger.info(f"Updating dropout values: dropout={dropout}, attention_dropout={attention_dropout}, activation_dropout={activation_dropout}")
        
        # Update encoder dropout
        for module in self.model.model.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        
        # Update decoder dropout
        for module in self.model.model.decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        
        # Update attention dropout
        if attention_dropout > 0.0:
            from transformers.models.whisper.modeling_whisper import WhisperAttention
            for module in self.model.modules():
                if isinstance(module, WhisperAttention):
                    module.dropout = attention_dropout
        
        # Update activation dropout
        if activation_dropout > 0.0:
            from transformers.activations import ACT2FN
            for name, module in self.model.named_modules():
                if "activation" in name and callable(module):
                    # This is a bit hacky, but we can wrap the activation function
                    # with a dropout layer
                    act_fn = module
                    setattr(self.model, name, lambda x: F.dropout(act_fn(x), p=activation_dropout, training=self.training))
    
    def _enable_flash_attention(self):
        """
        Enable flash attention for faster training if available.
        """
        try:
            from transformers.models.whisper.modeling_whisper import WhisperAttention
            
            # Check if flash attention is available
            if hasattr(nn.functional, 'scaled_dot_product_attention'):
                # Monkey patch the attention implementation
                def forward_flash_attn(self, query, key, value, attention_mask=None, head_mask=None):
                    # Reshape query, key, value for multi-head attention
                    batch_size = query.shape[0]
                    query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                    key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                    value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                    
                    # Apply flash attention
                    attn_output = nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0
                    )
                    
                    # Reshape output
                    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
                    
                    # Apply output projection
                    attn_output = self.out_proj(attn_output)
                    
                    return attn_output, None
                
                # Apply the patched forward method to all attention layers
                for module in self.model.modules():
                    if isinstance(module, WhisperAttention):
                        module.forward = forward_flash_attn.__get__(module, WhisperAttention)
                
                logger.info("Flash attention enabled for faster training.")
            else:
                logger.warning("Flash attention not available in this PyTorch version.")
        except Exception as e:
            logger.error(f"Failed to enable flash attention: {e}")
    
    def _replace_attention_with_relative_pos(self):
        """
        Replace standard attention with relative positional attention.
        """
        try:
            from transformers.models.whisper.modeling_whisper import WhisperAttention
            
            # Create new attention layers with relative positional encoding
            for name, module in self.model.named_modules():
                if isinstance(module, WhisperAttention):
                    # Create new attention layer
                    rel_pos_attn = MultiHeadAttentionWithRelPos(
                        embed_dim=module.embed_dim,
                        num_heads=module.num_heads,
                        dropout=module.dropout,
                        max_relative_position=64
                    )
                    
                    # Copy weights
                    rel_pos_attn.q_proj.weight.data.copy_(module.q_proj.weight.data)
                    rel_pos_attn.k_proj.weight.data.copy_(module.k_proj.weight.data)
                    rel_pos_attn.v_proj.weight.data.copy_(module.v_proj.weight.data)
                    rel_pos_attn.out_proj.weight.data.copy_(module.out_proj.weight.data)
                    
                    if module.q_proj.bias is not None:
                        rel_pos_attn.q_proj.bias.data.copy_(module.q_proj.bias.data)
                        rel_pos_attn.k_proj.bias.data.copy_(module.k_proj.bias.data)
                        rel_pos_attn.v_proj.bias.data.copy_(module.v_proj.bias.data)
                        rel_pos_attn.out_proj.bias.data.copy_(module.out_proj.bias.data)
                    
                    # Replace module
                    parent_name = name.rsplit(".", 1)[0]
                    child_name = name.rsplit(".", 1)[1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, rel_pos_attn)
            
            logger.info("Replaced standard attention with relative positional attention.")
        except Exception as e:
            logger.error(f"Failed to replace attention with relative positional attention: {e}")
    
    def _apply_spec_augment(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input features.
        
        Args:
            input_features: Input features of shape (batch_size, sequence_length, feature_dim)
            
        Returns:
            Augmented features
        """
        if not self.training or not self.use_spec_augment:
            return input_features
        
        # Reshape for SpecAugment (batch_size, channels, time, freq)
        batch_size, time, freq = input_features.size()
        features = input_features.unsqueeze(1)  # (batch_size, 1, time, freq)
        
        # Apply SpecAugment
        augmented = self.spec_augment(features)
        
        # Reshape back
        return augmented.squeeze(1)  # (batch_size, time, freq)
    
    def _apply_conformer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Conformer convolution modules to hidden states.
        
        Args:
            hidden_states: Hidden states from encoder
            
        Returns:
            Processed hidden states
        """
        if not self.use_conformer:
            return hidden_states
        
        # Apply Conformer modules
        for layer in self.conformer_layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states
    
    def _get_encoder_hook(self) -> Callable:
        """
        Get hook function for encoder to apply custom processing.
        
        Returns:
            Hook function
        """
        def encoder_hook(module, input, output):
            # Apply Conformer modules if enabled
            if self.use_conformer:
                output = self._apply_conformer(output)
            return output
        
        return encoder_hook
    
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict:
        """
        Forward pass of the model.
        
        Args:
            input_features: Audio features
            attention_mask: Attention mask for encoder
            decoder_input_ids: Input IDs for decoder
            decoder_attention_mask: Attention mask for decoder
            labels: Target labels
            
        Returns:
            Model outputs
        """
        # Apply SpecAugment if enabled
        if input_features is not None and self.use_spec_augment and self.training:
            input_features = self._apply_spec_augment(input_features)
        
        # Register hook for encoder if using Conformer
        if self.use_conformer:
            handle = self.model.model.encoder.register_forward_hook(self._get_encoder_hook())
        else:
            handle = None
        
        # Forward pass
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Remove hook
        if handle is not None:
            handle.remove()
        
        return outputs
    
    def generate(self, *args, **kwargs):
        """
        Generate text from audio features.
        
        Args:
            *args: Positional arguments for generation
            **kwargs: Keyword arguments for generation
            
        Returns:
            Generated text
        """
        return self.model.generate(*args, **kwargs)
    
    def save_pretrained(self, output_dir: str):
        """
        Save the mode
(Content truncated due to size limit. Use line ranges to read in chunks)