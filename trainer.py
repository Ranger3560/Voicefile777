"""
Training utilities for speech recognition models.

This module contains utilities for training speech recognition models,
including training loop, logging, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
from dataclasses import dataclass, field, asdict
import evaluate

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.whisper_model import WhisperModelWrapper, EnhancedWhisperModel
from utils.metrics import compute_wer, compute_cer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SpeechTrainingArguments:
    """
    Arguments for speech model training.
    """
    
    # Output settings
    output_dir: str = field(
        default="./output",
        metadata={"help": "Output directory for model checkpoints and logs"}
    )
    
    # Training settings
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before updating weights"}
    )
    
    # Optimizer settings
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay rate"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    
    # Scheduler settings
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "Learning rate scheduler type (linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup)"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warmup steps for learning rate scheduler"}
    )
    
    # Mixed precision settings
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training"}
    )
    
    # Logging settings
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for TensorBoard logs"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between logging updates"}
    )
    
    # Evaluation settings
    eval_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between evaluations"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy (no, steps, epoch)"}
    )
    
    # Saving settings
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    
    # Early stopping settings
    early_stopping_patience: Optional[int] = field(
        default=3,
        metadata={"help": "Number of evaluations with no improvement before early stopping"}
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.01,
        metadata={"help": "Minimum improvement required to reset early stopping counter"}
    )
    
    # Wandb settings
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name"}
    )
    
    # Miscellaneous settings
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    
    def to_transformers_args(self) -> Seq2SeqTrainingArguments:
        """
        Convert to Transformers Seq2SeqTrainingArguments.
        
        Returns:
            Seq2SeqTrainingArguments
        """
        # Create dictionary of arguments
        args_dict = asdict(self)
        
        # Remove arguments not supported by Seq2SeqTrainingArguments
        args_dict.pop("use_wandb", None)
        args_dict.pop("wandb_project", None)
        args_dict.pop("wandb_run_name", None)
        args_dict.pop("early_stopping_patience", None)
        args_dict.pop("early_stopping_threshold", None)
        
        # Create Seq2SeqTrainingArguments
        return Seq2SeqTrainingArguments(**args_dict)
    
    def save_to_json(self, output_file: str) -> str:
        """
        Save arguments to JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        with open(output_file, "w") as f:
            json.dump(asdict(self), f, indent=2)
        
        return output_file
    
    @classmethod
    def from_json(cls, json_file: str) -> "SpeechTrainingArguments":
        """
        Load arguments from JSON file.
        
        Args:
            json_file: JSON file path
            
        Returns:
            SpeechTrainingArguments
        """
        with open(json_file, "r") as f:
            args_dict = json.load(f)
        
        return cls(**args_dict)


class SpeechTrainer:
    """
    Trainer for speech recognition models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: SpeechTrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        processor: Optional[WhisperProcessor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_metrics: Optional[Callable] = None
    ):
        """
        Initialize speech trainer.
        
        Args:
            model: Speech recognition model
            args: Training arguments
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            processor: Whisper processor
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            compute_metrics: Function to compute metrics
        """
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.processor = processor
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Set up learning rate scheduler
        if lr_scheduler is None:
            num_training_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
            
            if args.lr_scheduler_type == "linear":
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=num_training_steps
                )
            elif args.lr_scheduler_type == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_training_steps
                )
            elif args.lr_scheduler_type == "cosine_with_restarts":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=num_training_steps // 3,
                    T_mult=2
                )
            elif args.lr_scheduler_type == "polynomial":
                self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                    self.optimizer,
                    total_iters=num_training_steps,
                    power=1.0
                )
            elif args.lr_scheduler_type == "constant":
                self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    factor=1.0,
                    total_iters=num_training_steps
                )
            elif args.lr_scheduler_type == "constant_with_warmup":
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=args.warmup_steps
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler_type}")
        else:
            self.lr_scheduler = lr_scheduler
        
        # Set up metrics function
        if compute_metrics is None and processor is not None:
            self.compute_metrics = self._default_compute_metrics
        else:
            self.compute_metrics = compute_metrics
        
        # Set up logging
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        self.no_improvement_count = 0
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set up TensorBoard
        if args.logging_dir is not None:
            self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
        else:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
        
        # Set up Weights & Biases
        if args.use_wandb:
            wandb_config = {
                "model_name": model.__class__.__name__,
                **asdict(args)
            }
            
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=wandb_config
            )
        
        # Save training arguments
        self.args.save_to_json(os.path.join(args.output_dir, "training_args.json"))
        
        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    def _default_compute_metrics(self, pred_ids: List[List[int]], label_ids: List[List[int]]) -> Dict[str, float]:
        """
        Default function to compute metrics.
        
        Args:
            pred_ids: Predicted token IDs
            label_ids: Label token IDs
            
        Returns:
            Dictionary of metrics
        """
        # Replace -100 with pad token ID
        label_ids = [[id if id != -100 else self.processor.tokenizer.pad_token_id for id in ids] for ids in label_ids]
        
        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = compute_wer(pred_str, label_str)
        
        # Compute CER
        cer = compute_cer(pred_str, label_str)
        
        return {
            "wer": wer,
            "cer": cer
        }
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Starting training")
        
        # Set model to training mode
        self.model.train()
        
        # Initialize progress bar
        progress_bar = tqdm(total=len(self.train_dataloader) * self.args.num_train_epochs)
        
        # Initialize metrics
        train_losses = []
        eval_metrics = {}
        
        # Training loop
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if gradient accumulation is complete
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Log metrics
                    if self.global_step % self.args.logging_steps == 0:
                        # Log to TensorBoard
                        self.tb_writer.add_scalar("train/loss", loss.item() * self.args.gradient_accumulation_steps, self.global_step)
                        self.tb_writer.add_scalar("train/learning_rate", self.lr_scheduler.get_last_lr()[0], self.global_step)
                        
                        # Log to Weights & Biases
                        if self.args.use_wandb:
                            wandb.log({
                                "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                                "train/epoch": self.epoch,
                                "train/global_step": self.global_step
                            })
                    
                    # Evaluate model
                    if self.eval_dataloader is not None and self.args.evaluation_strategy == "steps" and self.global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate()
                        eval_metrics[self.global_step] = metrics
                        
                        # Check for early stopping
                        if self._check_early_stopping(metrics):
                            logger.info("Early stopping triggered")
                            return {
                                "train_loss": np.mean(train_losses),
                                **metrics
                            }
                    
                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(metrics=eval_metrics.get(self.global_step))
                
  
(Content truncated due to size limit. Use line ranges to read in chunks)