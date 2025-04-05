"""
Whisper model configuration utilities for speech recognition.

This module contains functions for configuring and customizing Whisper models
for speech recognition tasks, including model size selection and parameter tuning.
"""

import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperModelConfig:
    """
    Configuration class for Whisper models.
    """
    
    # Default configurations for different model sizes
    MODEL_SIZES = {
        "tiny": {
            "d_model": 384,
            "encoder_layers": 4,
            "encoder_attention_heads": 6,
            "decoder_layers": 4,
            "decoder_attention_heads": 6,
            "encoder_ffn_dim": 1536,
            "decoder_ffn_dim": 1536
        },
        "base": {
            "d_model": 512,
            "encoder_layers": 6,
            "encoder_attention_heads": 8,
            "decoder_layers": 6,
            "decoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "decoder_ffn_dim": 2048
        },
        "small": {
            "d_model": 768,
            "encoder_layers": 12,
            "encoder_attention_heads": 12,
            "decoder_layers": 12,
            "decoder_attention_heads": 12,
            "encoder_ffn_dim": 3072,
            "decoder_ffn_dim": 3072
        },
        "medium": {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "decoder_layers": 24,
            "decoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096
        },
        "large": {
            "d_model": 1280,
            "encoder_layers": 32,
            "encoder_attention_heads": 20,
            "decoder_layers": 32,
            "decoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "decoder_ffn_dim": 5120
        }
    }
    
    # Multilingual model configurations
    MULTILINGUAL_CONFIGS = {
        "en": {"vocab_size": 51865, "language": "english"},
        "multilingual": {"vocab_size": 51865, "language": None}
    }
    
    def __init__(
        self,
        model_size: str = "tiny",
        multilingual: bool = False,
        language: Optional[str] = None,
        use_flash_attention: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        use_cache: bool = True,
        activation_function: str = "gelu",
        d_model: Optional[int] = None,
        encoder_layers: Optional[int] = None,
        encoder_attention_heads: Optional[int] = None,
        decoder_layers: Optional[int] = None,
        decoder_attention_heads: Optional[int] = None,
        encoder_ffn_dim: Optional[int] = None,
        decoder_ffn_dim: Optional[int] = None,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize Whisper model configuration.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            multilingual: Whether to use multilingual model
            language: Language for multilingual model
            use_flash_attention: Whether to use flash attention
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            activation_dropout: Activation dropout probability
            encoder_layerdrop: Encoder layer dropout probability
            decoder_layerdrop: Decoder layer dropout probability
            use_cache: Whether to use cache for faster inference
            activation_function: Activation function
            d_model: Model dimension (overrides model_size defaults)
            encoder_layers: Number of encoder layers (overrides model_size defaults)
            encoder_attention_heads: Number of encoder attention heads (overrides model_size defaults)
            decoder_layers: Number of decoder layers (overrides model_size defaults)
            decoder_attention_heads: Number of decoder attention heads (overrides model_size defaults)
            encoder_ffn_dim: Encoder FFN dimension (overrides model_size defaults)
            decoder_ffn_dim: Decoder FFN dimension (overrides model_size defaults)
            vocab_size: Vocabulary size (overrides defaults)
        """
        # Validate model size
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(self.MODEL_SIZES.keys())}")
        
        # Get default configuration for the specified model size
        self.config_dict = self.MODEL_SIZES[model_size].copy()
        
        # Set multilingual configuration
        if multilingual:
            self.config_dict.update(self.MULTILINGUAL_CONFIGS["multilingual"])
        else:
            self.config_dict.update(self.MULTILINGUAL_CONFIGS["en"])
        
        # Set language if specified
        if language is not None:
            self.config_dict["language"] = language
        
        # Set dropout values
        self.config_dict["dropout"] = dropout
        self.config_dict["attention_dropout"] = attention_dropout
        self.config_dict["activation_dropout"] = activation_dropout
        self.config_dict["encoder_layerdrop"] = encoder_layerdrop
        self.config_dict["decoder_layerdrop"] = decoder_layerdrop
        
        # Set other configuration values
        self.config_dict["use_cache"] = use_cache
        self.config_dict["activation_function"] = activation_function
        self.config_dict["use_flash_attention"] = use_flash_attention
        
        # Override defaults with specified values
        if d_model is not None:
            self.config_dict["d_model"] = d_model
        
        if encoder_layers is not None:
            self.config_dict["encoder_layers"] = encoder_layers
        
        if encoder_attention_heads is not None:
            self.config_dict["encoder_attention_heads"] = encoder_attention_heads
        
        if decoder_layers is not None:
            self.config_dict["decoder_layers"] = decoder_layers
        
        if decoder_attention_heads is not None:
            self.config_dict["decoder_attention_heads"] = decoder_attention_heads
        
        if encoder_ffn_dim is not None:
            self.config_dict["encoder_ffn_dim"] = encoder_ffn_dim
        
        if decoder_ffn_dim is not None:
            self.config_dict["decoder_ffn_dim"] = decoder_ffn_dim
        
        if vocab_size is not None:
            self.config_dict["vocab_size"] = vocab_size
    
    def to_transformers_config(self) -> WhisperConfig:
        """
        Convert to Transformers WhisperConfig.
        
        Returns:
            WhisperConfig object
        """
        return WhisperConfig(**self.config_dict)
    
    def save_to_json(self, output_file: str) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        with open(output_file, "w") as f:
            json.dump(self.config_dict, f, indent=2)
        
        return output_file
    
    @classmethod
    def from_json(cls, json_file: str) -> "WhisperModelConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_file: JSON file path
            
        Returns:
            WhisperModelConfig object
        """
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        
        # Extract model size
        model_size = "tiny"  # Default
        for size, size_config in cls.MODEL_SIZES.items():
            if config_dict.get("d_model") == size_config["d_model"] and \
               config_dict.get("encoder_layers") == size_config["encoder_layers"]:
                model_size = size
                break
        
        # Create instance
        instance = cls(model_size=model_size)
        
        # Update with loaded values
        instance.config_dict.update(config_dict)
        
        return instance
    
    def create_model(self) -> WhisperForConditionalGeneration:
        """
        Create a Whisper model with this configuration.
        
        Returns:
            WhisperForConditionalGeneration model
        """
        config = self.to_transformers_config()
        return WhisperForConditionalGeneration(config)
    
    def __str__(self) -> str:
        """
        String representation of the configuration.
        
        Returns:
            String representation
        """
        return f"WhisperModelConfig({json.dumps(self.config_dict, indent=2)})"


def create_whisper_model(
    model_name_or_path: str = "openai/whisper-tiny",
    use_pretrained: bool = True,
    custom_config: Optional[WhisperModelConfig] = None,
    device: Optional[str] = None
) -> WhisperForConditionalGeneration:
    """
    Create a Whisper model.
    
    Args:
        model_name_or_path: Model name or path
        use_pretrained: Whether to use pretrained weights
        custom_config: Custom model configuration
        device: Device to load model on
        
    Returns:
        WhisperForConditionalGeneration model
    """
    if custom_config is not None:
        # Create model from custom configuration
        model = custom_config.create_model()
        
        if use_pretrained:
            # Load pretrained weights if available
            try:
                pretrained_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
                
                # Copy compatible weights
                pretrained_state_dict = pretrained_model.state_dict()
                model_state_dict = model.state_dict()
                
                for name, param in pretrained_state_dict.items():
                    if name in model_state_dict and param.shape == model_state_dict[name].shape:
                        model_state_dict[name].copy_(param)
                
                logger.info(f"Loaded compatible pretrained weights from {model_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
    else:
        # Load model from pretrained or create from scratch
        if use_pretrained:
            model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        else:
            config = WhisperConfig.from_pretrained(model_name_or_path)
            model = WhisperForConditionalGeneration(config)
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    return model


def create_distilled_whisper_model(
    teacher_model_path: str,
    student_config: WhisperModelConfig,
    device: Optional[str] = None
) -> WhisperForConditionalGeneration:
    """
    Create a distilled Whisper model.
    
    Args:
        teacher_model_path: Path to teacher model
        student_config: Configuration for student model
        device: Device to load model on
        
    Returns:
        Distilled WhisperForConditionalGeneration model
    """
    # Load teacher model
    teacher_model = WhisperForConditionalGeneration.from_pretrained(teacher_model_path)
    
    # Create student model
    student_model = student_config.create_model()
    
    # Initialize student with teacher's encoder
    # This is a simple form of distillation by weight initialization
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = student_model.state_dict()
    
    # Copy encoder weights where shapes match
    for name, param in teacher_state_dict.items():
        if name.startswith("model.encoder") and name in student_state_dict:
            if param.shape == student_state_dict[name].shape:
                student_state_dict[name].copy_(param)
    
    # Move model to device if specified
    if device is not None:
        student_model = student_model.to(device)
    
    return student_model


def save_model_config(
    model: WhisperForConditionalGeneration,
    output_dir: str,
    config_name: str = "config.json"
) -> str:
    """
    Save model configuration.
    
    Args:
        model: Whisper model
        output_dir: Output directory
        config_name: Configuration file name
        
    Returns:
        Path to saved configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, config_name)
    
    # Save configuration
    model.config.to_json_file(output_file)
    
    return output_file


def compare_model_configs(
    config1: Union[WhisperConfig, WhisperModelConfig],
    config2: Union[WhisperConfig, WhisperModelConfig]
) -> Dict[str, Any]:
    """
    Compare two model configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary of differences
    """
    # Convert to dictionaries
    if isinstance(config1, WhisperModelConfig):
        dict1 = config1.config_dict
    else:
        dict1 = config1.to_dict()
    
    if isinstance(config2, WhisperModelConfig):
        dict2 = config2.config_dict
    else:
        dict2 = config2.to_dict()
    
    # Find differences
    differences = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        if key not in dict1:
            differences[key] = (None, dict2[key])
        elif key not in dict2:
            differences[key] = (dict1[key], None)
        elif dict1[key] != dict2[key]:
            differences[key] = (dict1[key], dict2[key])
    
    return differences


def get_model_size_info(config: Union[WhisperConfig, WhisperModelConfig]) -> Dict[str, Any]:
    """
    Get information about model size.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary of model size information
    """
    # Convert to dictionary
    if isinstance(config, WhisperModelConfig):
        config_dict = config.config_dict
    else:
        config_dict = config.to_dict()
    
    # Calculate number of parameters
    encoder_params = config_dict["encoder_layers"] * (
        4 * config_dict["d_model"] * config_dict["encoder_ffn_dim"] +  # FFN
        4 * config_dict["d_model"] * config_dict["d_model"]  # Self-attention
    )
    
    decoder_params = config_dict["decoder_layers"] * (
        4 * config_dict["d_model"] * config_dict["decoder_ffn_dim"] +  # FFN
        4 * config_dict["d_model"] * config_dict["d_model"] +  # Self-attention
        4 * config_dict["d_model"] * config_dict["d_model"]  # Cross-attention
    )
    
    embedding_params = config_dict["vocab_size"] * config_dict["d_model"]
    
    total_params = encoder_params + decoder_params + embedding_params
    
    # Determine model size category
    model_size = "custom"
    for size, size_config in WhisperModelConfig.MODEL_SIZES.items():
        if config_dict.get("d_model") == size_config["d_model"] and \
           config_dict.get("encoder_layers") == size_config["encoder_layers"]:
            model_size = size
            break
    
    return {
        "model_size": model_size,
        "d_model": config_dict["d_model"],
        "encoder_layers": config_dict["encoder_layers"],
        "decoder_layers": config_dict["decoder_layers"],
        "encoder_attention_heads": config_dict["encoder_attention_heads"],
        "decoder_attention_heads": config_dict["decoder_attention_heads"],
        "encoder_ffn_dim": config_dict["encoder_ffn_dim"],
        "decoder_ffn_dim": config_dict["decoder_ffn_dim"],
        "vocab_size": config_dict["vocab_s
(Content truncated due to size limit. Use line ranges to read in chunks)