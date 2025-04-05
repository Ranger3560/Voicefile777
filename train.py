"""
Training script for speech recognition models.

This script provides a command-line interface for training speech recognition models
using the SpeechTrainer class.
"""

import argparse
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader
import json

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.trainer import SpeechTrainingArguments, SpeechTrainer, create_compute_metrics_fn
from models.whisper_model import create_whisper_model_and_processor
from data.librispeech import create_librispeech_dataloaders, create_data_collator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a speech recognition model")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-tiny",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--use_enhanced_model",
        action="store_true",
        help="Whether to use the enhanced model with custom components"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze the encoder parameters"
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        help="Whether to freeze the decoder parameters"
    )
    parser.add_argument(
        "--use_spec_augment",
        action="store_true",
        help="Whether to use SpecAugment for data augmentation"
    )
    parser.add_argument(
        "--use_conformer",
        action="store_true",
        help="Whether to use Conformer convolution modules"
    )
    parser.add_argument(
        "--use_relative_positional_encoding",
        action="store_true",
        help="Whether to use relative positional encoding"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="librispeech_asr",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="clean",
        help="Dataset configuration"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train.100",
        help="Training split"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation.clean",
        help="Evaluation split"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for model checkpoints and logs"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps between logging updates"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between evaluations"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between saving checkpoints"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement before early stopping"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.01,
        help="Minimum improvement required to reset early stopping counter"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="speech-recognition",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    return args


def main():
    """
    Main function for training a speech recognition model.
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create model and processor
    logger.info(f"Creating model: {args.model_name_or_path}")
    model, processor = create_whisper_model_and_processor(
        model_name_or_path=args.model_name_or_path,
        use_enhanced_model=args.use_enhanced_model,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        use_spec_augment=args.use_spec_augment,
        use_conformer=args.use_conformer,
        use_relative_positional_encoding=args.use_relative_positional_encoding
    )
    
    # Create data collator
    data_collator = create_data_collator(processor)
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    max_samples = {}
    if args.max_train_samples is not None:
        max_samples[args.train_split] = args.max_train_samples
    if args.max_eval_samples is not None:
        max_samples[args.eval_split] = args.max_eval_samples
    
    dataloaders = create_librispeech_dataloaders(
        train_split=args.train_split,
        eval_split=args.eval_split,
        config=args.dataset_config,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        max_samples=max_samples
    )
    
    train_dataloader = dataloaders["train"]
    eval_dataloader = dataloaders["eval"]
    
    # Create training arguments
    training_args = SpeechTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed
    )
    
    # Create compute metrics function
    compute_metrics = create_compute_metrics_fn(processor)
    
    # Create trainer
    trainer = SpeechTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        processor=processor,
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info("Starting training")
    metrics = trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    else:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model", "model.pt"))
    
    # Save processor
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Save final metrics
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
