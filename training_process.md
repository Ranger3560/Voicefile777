# Training Process Documentation

This document provides a detailed overview of the training process for the speech recognition model. It covers data preparation, training configuration, monitoring, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Training Configuration](#training-configuration)
4. [Training Loop](#training-loop)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Early Stopping](#early-stopping)
7. [Checkpointing](#checkpointing)
8. [Best Practices](#best-practices)

## Overview

Training a speech recognition model involves several steps, from data preparation to model evaluation. Our training pipeline is designed to be flexible, efficient, and easy to use, with support for various training configurations and monitoring tools.

The training process is implemented in the `scripts/train.py` and `scripts/trainer.py` files, which provide a command-line interface and a robust training framework.

## Data Preparation

### Dataset Loading

We use the LibriSpeech dataset for training, which is a large corpus of read English speech. The dataset is loaded using the Hugging Face Datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("librispeech_asr", "clean", split="train.100")
```

### Preprocessing

The audio data is preprocessed to ensure consistent format and quality:

1. **Resampling**: All audio is resampled to 16kHz
2. **Feature Extraction**: Log-mel spectrograms are computed with 80 mel bins
3. **Normalization**: Features are normalized to zero mean and unit variance
4. **Tokenization**: Text is tokenized using the Whisper tokenizer

```python
def preprocess(batch):
    # Resample audio to 16kHz
    audio_array = librosa.resample(
        batch["audio"]["array"], 
        orig_sr=batch["audio"]["sampling_rate"], 
        target_sr=16000
    )
    
    # Extract features
    features = extract_features(audio_array)
    
    # Tokenize text
    tokens = processor.tokenizer(batch["text"])
    
    return {"input_features": features, "labels": tokens.input_ids}
```

### Data Augmentation

We apply several data augmentation techniques to improve model robustness:

1. **Time Stretching**: Randomly speed up or slow down the audio
2. **Pitch Shifting**: Randomly shift the pitch up or down
3. **Noise Addition**: Add random noise to the audio
4. **Room Impulse Response**: Simulate different acoustic environments

```python
def augment_audio(audio, sample_rate=16000):
    # Time stretching
    stretch_factor = np.random.uniform(0.8, 1.2)
    audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    # Pitch shifting
    n_steps = np.random.randint(-3, 4)
    audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    
    # Add noise
    noise_level = np.random.uniform(0, 0.005)
    noise = np.random.randn(len(audio))
    audio = audio + noise_level * noise
    
    return audio
```

### Data Streaming

For efficient training on large datasets, we use streaming mode to avoid loading the entire dataset into memory:

```python
dataset = dataset.map(preprocess, remove_columns=["audio"]).with_format("torch")
```

## Training Configuration

### Training Arguments

The training process is configured using the `SpeechTrainingArguments` class, which provides a wide range of options:

```python
training_args = SpeechTrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    warmup_steps=500,
    fp16=True,
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    save_total_limit=3,
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)
```

### Model Configuration

The model is configured using the `WhisperModelConfig` class or by directly specifying parameters:

```python
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",
    use_enhanced_model=True,
    freeze_encoder=False,
    freeze_decoder=False,
    use_spec_augment=True,
    use_conformer=True,
    use_relative_positional_encoding=True
)
```

### Optimizer and Scheduler

We use the AdamW optimizer with a linear learning rate scheduler:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

num_training_steps = len(train_dataloader) * args.num_train_epochs
lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.0,
    total_iters=num_training_steps
)
```

## Training Loop

The training loop is implemented in the `SpeechTrainer` class, which handles the entire training process:

```python
trainer = SpeechTrainer(
    model=model,
    args=training_args,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    processor=processor
)

metrics = trainer.train()
```

The training loop includes the following steps:

1. **Forward Pass**: Compute model outputs and loss
2. **Backward Pass**: Compute gradients
3. **Gradient Accumulation**: Accumulate gradients over multiple batches
4. **Gradient Clipping**: Clip gradients to prevent exploding gradients
5. **Weight Update**: Update model weights using the optimizer
6. **Learning Rate Update**: Update learning rate using the scheduler
7. **Logging**: Log metrics to TensorBoard and/or Weights & Biases
8. **Evaluation**: Periodically evaluate the model on the validation set
9. **Checkpointing**: Save model checkpoints

## Monitoring and Logging

### TensorBoard

We use TensorBoard for monitoring training progress:

```python
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter(log_dir=args.logging_dir)
tb_writer.add_scalar("train/loss", loss.item(), global_step)
tb_writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
```

### Weights & Biases

We also support Weights & Biases for more advanced experiment tracking:

```python
import wandb

wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config=wandb_config
)

wandb.log({
    "train/loss": loss.item(),
    "train/learning_rate": lr_scheduler.get_last_lr()[0],
    "train/epoch": epoch,
    "train/global_step": global_step
})
```

### Console Logging

Progress is also logged to the console using the Python logging module and tqdm progress bars:

```python
import logging
from tqdm import tqdm

logging.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Loss: {epoch_loss:.4f}")
progress_bar = tqdm(total=len(train_dataloader) * args.num_train_epochs)
progress_bar.update(1)
```

## Early Stopping

We implement early stopping to prevent overfitting and save training time:

```python
def _check_early_stopping(self, metrics):
    if self.args.early_stopping_patience is None:
        return False
    
    # Get metric for early stopping (default to loss)
    metric_name = "wer" if "wer" in metrics else "loss"
    metric_value = metrics[metric_name]
    
    # Check if metric improved
    if metric_value < self.best_metric - self.args.early_stopping_threshold:
        # Metric improved
        self.best_metric = metric_value
        self.no_improvement_count = 0
        return False
    else:
        # Metric did not improve
        self.no_improvement_count += 1
        
        # Check if patience is exceeded
        if self.no_improvement_count >= self.args.early_stopping_patience:
            return True
        else:
            return False
```

## Checkpointing

We save model checkpoints during training to allow resuming training and to keep the best models:

```python
def _save_checkpoint(self, metrics=None):
    # Create checkpoint directory
    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    if hasattr(self.model, "save_pretrained"):
        self.model.save_pretrained(checkpoint_dir)
    else:
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
    
    # Save processor
    if self.processor is not None and hasattr(self.processor, "save_pretrained"):
        self.processor.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler
    torch.save(
        {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step
        },
        os.path.join(checkpoint_dir, "optimizer.pt")
    )
    
    # Save metrics
    if metrics is not None:
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
```

We also implement checkpoint rotation to limit disk usage:

```python
def _rotate_checkpoints(self):
    # Get all checkpoint directories
    checkpoint_dirs = [
        d for d in os.listdir(self.args.output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.args.output_dir, d))
    ]
    
    # Sort by step number
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda x: int(x.split("-")[1])
    )
    
    # Remove oldest checkpoints if limit is exceeded
    if len(checkpoint_dirs) > self.args.save_total_limit:
        num_to_remove = len(checkpoint_dirs) - self.args.save_total_limit
        
        for checkpoint_dir in checkpoint_dirs[:num_to_remove]:
            logger.info(f"Removing old checkpoint: {checkpoint_dir}")
            import shutil
            shutil.rmtree(os.path.join(self.args.output_dir, checkpoint_dir))
```

## Best Practices

### Hyperparameter Tuning

For best results, we recommend tuning the following hyperparameters:

- **Learning Rate**: Start with 5e-5 and adjust based on validation loss
- **Batch Size**: Use the largest batch size that fits in memory
- **Gradient Accumulation Steps**: Increase for larger effective batch sizes
- **Weight Decay**: 0.01 is a good starting point
- **Warmup Steps**: 10% of total training steps is a good rule of thumb

### Mixed Precision Training

We recommend using mixed precision training (FP16) for faster training:

```python
training_args = SpeechTrainingArguments(
    # ...
    fp16=True,
    # ...
)
```

### Gradient Clipping

To prevent exploding gradients, we recommend using gradient clipping:

```python
training_args = SpeechTrainingArguments(
    # ...
    max_grad_norm=1.0,
    # ...
)
```

### Data Augmentation

For best results, we recommend using data augmentation:

```python
model, processor = create_whisper_model_and_processor(
    # ...
    use_spec_augment=True,
    # ...
)
```

### Model Freezing

For fine-tuning on small datasets, consider freezing the encoder:

```python
model, processor = create_whisper_model_and_processor(
    # ...
    freeze_encoder=True,
    # ...
)
```

### Distributed Training

For large models and datasets, consider using distributed training:

```bash
python -m torch.distributed.launch --nproc_per_node=8 scripts/train.py \
    --model_name_or_path="openai/whisper-small" \
    --output_dir="./output" \
    --num_train_epochs=3 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --fp16
```
