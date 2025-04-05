# Google Colab Setup Instructions

This document provides detailed instructions for setting up and running the speech recognition model in Google Colab. These instructions are designed to help you get started quickly without requiring local disk space or GPU resources.

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Model Optimization](#model-optimization)
8. [Troubleshooting](#troubleshooting)

## Overview

Google Colab provides a free cloud environment with GPU support, making it ideal for training and evaluating speech recognition models. This guide will walk you through the process of setting up the environment, preparing the data, and running the model in Google Colab.

## Environment Setup

### Step 1: Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Rename the notebook to "Speech Recognition Model"

### Step 2: Set Up GPU Acceleration

1. Click on "Runtime" in the menu
2. Select "Change runtime type"
3. Set "Hardware accelerator" to "GPU"
4. Click "Save"

### Step 3: Clone the Repository

Run the following code in a Colab cell to clone the repository:

```python
# Clone the repository
!git clone https://github.com/your-username/speech_model_project.git
%cd speech_model_project
```

### Step 4: Install Dependencies

Run the following code to install the required dependencies:

```python
# Install dependencies
!pip install torch torchaudio transformers datasets librosa soundfile evaluate jiwer tensorboard matplotlib numpy pandas
```

## Project Structure

The project has the following structure:

```
speech_model_project/
├── data/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── librispeech.py
│   └── utils.py
├── models/
│   ├── whisper_model.py
│   ├── components.py
│   ├── config.py
│   └── custom_layers.py
├── scripts/
│   ├── train.py
│   ├── trainer.py
│   ├── evaluate.py
│   └── optimize.py
├── utils/
│   ├── metrics.py
│   ├── optimization.py
│   └── visualization.py
├── notebooks/
│   └── demo.ipynb
└── docs/
    ├── model_architecture.md
    ├── training_process.md
    ├── evaluation_methodology.md
    └── api_documentation.md
```

## Data Preparation

### Step 1: Import Libraries

```python
import torch
import torchaudio
from datasets import load_dataset
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperProcessor
import sys
import os

# Add project directory to path
sys.path.append('/content/speech_model_project')

# Import project modules
from data.preprocessing import preprocess_audio, extract_features
from data.augmentation import augment_audio
from data.librispeech import create_librispeech_dataloaders, create_data_collator
```

### Step 2: Load and Explore the Dataset

```python
# Load a sample from LibriSpeech
dataset = load_dataset("librispeech_asr", "clean", split="validation.clean[:10]")

# Display dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Dataset features: {dataset.features}")
print(f"Sample: {dataset[0]}")

# Play audio sample
from IPython.display import Audio
sample = dataset[0]
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])

# Visualize audio waveform
plt.figure(figsize=(10, 4))
plt.plot(sample["audio"]["array"])
plt.title("Audio Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()
```

### Step 3: Preprocess Data

```python
# Preprocess audio sample
audio_processed = preprocess_audio(
    sample["audio"]["array"],
    sample_rate=sample["audio"]["sampling_rate"],
    target_sample_rate=16000,
    normalize=True,
    trim_silence=True
)

# Extract features
features = extract_features(
    audio_processed,
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
    hop_length=160
)

# Visualize mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(features, aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Bin")
plt.colorbar(format='%+2.0f dB')
plt.show()
```

### Step 4: Create Data Loaders

```python
# Load processor
from models.whisper_model import create_whisper_model_and_processor
_, processor = create_whisper_model_and_processor(model_name_or_path="openai/whisper-tiny")

# Create data collator
data_collator = create_data_collator(processor)

# Create dataloaders
dataloaders = create_librispeech_dataloaders(
    train_split="train.100",
    eval_split="validation.clean",
    config="clean",
    batch_size=8,
    collate_fn=data_collator,
    max_samples={"train.100": 100, "validation.clean": 10}  # Limit samples for demonstration
)

# Get train and eval dataloaders
train_dataloader = dataloaders["train"]
eval_dataloader = dataloaders["eval"]

# Display dataloader information
print(f"Train dataloader size: {len(train_dataloader)}")
print(f"Eval dataloader size: {len(eval_dataloader)}")
```

## Model Training

### Step 1: Load Model

```python
from models.whisper_model import create_whisper_model_and_processor

# Load model and processor
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",
    use_enhanced_model=True,
    use_spec_augment=True,
    use_conformer=True,
    use_relative_positional_encoding=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Step 2: Configure Training

```python
from scripts.trainer import SpeechTrainingArguments, SpeechTrainer

# Create output directory
output_dir = "/content/output"
os.makedirs(output_dir, exist_ok=True)

# Create training arguments
training_args = SpeechTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    warmup_steps=100,
    fp16=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)
```

### Step 3: Train Model

```python
# Create trainer
trainer = SpeechTrainer(
    model=model,
    args=training_args,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    processor=processor
)

# Train model
metrics = trainer.train()

# Print training metrics
print(f"Training metrics: {metrics}")

# Save final model
model.save_pretrained(os.path.join(output_dir, "final_model"))
processor.save_pretrained(os.path.join(output_dir, "final_model"))
```

### Step 4: Visualize Training Results

```python
# Load TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/output/logs

# Plot training loss
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from TensorBoard
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(os.path.join(output_dir, "logs"))
ea.Reload()

# Extract loss values
loss_values = []
for event in ea.Scalars("train/loss"):
    loss_values.append((event.step, event.value))

# Create DataFrame
df = pd.DataFrame(loss_values, columns=["step", "loss"])

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["loss"])
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

## Model Evaluation

### Step 1: Load Trained Model

```python
from models.whisper_model import create_whisper_model_and_processor

# Load model and processor
model_path = "/content/output/final_model"
model, processor = create_whisper_model_and_processor(
    model_name_or_path=model_path,
    use_enhanced_model=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
```

### Step 2: Evaluate Model

```python
from utils.metrics import evaluate_model, create_evaluation_report

# Create evaluation directory
eval_dir = "/content/evaluation_results"
os.makedirs(eval_dir, exist_ok=True)

# Load test dataset
test_dataset = load_dataset("librispeech_asr", "clean", split="test.clean[:20]")

# Process test dataset
all_predictions = []
all_references = []

for sample in test_dataset:
    # Preprocess audio
    audio_processed = preprocess_audio(
        sample["audio"]["array"],
        sample_rate=sample["audio"]["sampling_rate"],
        target_sample_rate=16000,
        normalize=True,
        trim_silence=True
    )
    
    # Convert to tensor
    input_features = processor(audio_processed, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        output = model.generate(input_features)
    
    # Decode output
    prediction = processor.batch_decode(output, skip_special_tokens=True)[0]
    reference = sample["text"]
    
    # Store prediction and reference
    all_predictions.append(prediction)
    all_references.append(reference)

# Evaluate model
metrics = evaluate_model(
    all_predictions,
    all_references,
    output_dir=eval_dir
)

# Create evaluation report
report_path = create_evaluation_report(
    metrics,
    output_file=os.path.join(eval_dir, "evaluation_report.md")
)

# Print summary metrics
print(f"Word Error Rate (WER): {metrics['wer']:.4f}")
print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
print(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
print(f"Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
```

### Step 3: Visualize Evaluation Results

```python
# Display evaluation visualizations
from IPython.display import Image, display, Markdown

# Display WER by length chart
display(Image(os.path.join(eval_dir, "wer_by_length.png")))

# Display detailed WER metrics chart
display(Image(os.path.join(eval_dir, "detailed_wer_metrics.png")))

# Display confusion matrix
display(Image(os.path.join(eval_dir, "confusion_matrix.png")))

# Display evaluation report
with open(os.path.join(eval_dir, "evaluation_report.md"), "r") as f:
    report_content = f.read()
display(Markdown(report_content))
```

## Model Optimization

### Step 1: Quantize Model

```python
from utils.optimization import apply_dynamic_quantization, benchmark_model

# Apply quantization
quantized_model = apply_dynamic_quantization(model)

# Define input shape
input_shape = (1, 80, 3000)

# Benchmark original model
original_benchmark = benchmark_model(
    model=model,
    input_shape=input_shape,
    device=device,
    num_iterations=10,
    warmup_iterations=2
)

# Benchmark quantized model
quantized_benchmark = benchmark_model(
    model=quantized_model,
    input_shape=input_shape,
    device=device,
    num_iterations=10,
    warmup_iterations=2
)

# Print benchmark results
print("Benchmark Results:")
print(f"Original Model: {original_benchmark['latency_ms']:.2f} ms")
print(f"Quantized Model: {quantized_benchmark['latency_ms']:.2f} ms")
print(f"Speedup: {original_benchmark['latency_ms'] / quantized_benchmark['latency_ms']:.2f}x")
```

### Step 2: Export to ONNX

```python
from utils.optimization import export_to_onnx, benchmark_onnx_model

# Create optimization directory
opt_dir = "/content/optimized_models"
os.makedirs(opt_dir, exist_ok=True)

# Export to ONNX
onnx_path = export_to_onnx(
    model=model,
    output_path=os.path.join(opt_dir, "model.onnx"),
    input_shape=input_shape
)

# Benchmark ONNX model
onnx_benchmark = benchmark_onnx_model(
    onnx_path=onnx_path,
    input_shape=input_shape,
    num_iterations=10,
    warmup_iterations=2
)

# Print benchmark results
print(f"ONNX Model: {onnx_benchmark['latency_ms']:.2f} ms")
print(f"Speedup vs Original: {original_benchmark['latency_ms'] / onnx_benchmark['latency_ms']:.2f}x")
```

### Step 3: Save Optimized Model

```python
# Save quantized model
quantized_model_path = os.path.join(opt_dir, "quantized_model")
os.makedirs(quantized_model_path, exist_ok=True)

if hasattr(quantized_model, "save_pretrained"):
    quantized_model.save_pretrained(quantized_model_path)
else:
    torch.save(quantized_model.state_dict(), os.path.join(quantized_model_path, "model.pt"))

# Save processor
processor.save_pretrained(quantized_model_path)

print(f"Quantized model saved to {quantized_model_path}")
print(f"ONNX model saved to {onnx_path}")
```

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors

If you encounter out of memory errors, try the following:

1. Reduce batch size:
```python
training_args = SpeechTrainingArguments(
    per_device_train_batch_size=4,  # Reduce from 8 to 4
    # ...
)
```

2. Use gradient accumulation:
```python
training_args = SpeechTrainingArguments(
    gradient_accumulation_steps=4,  # Increase from 2 to 4
    # ...
)
```

3. Use a smaller model:
```python
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",  # Use tiny instead of base or larger
    # ...
)
```

#### Slow Training

If training is too slow, try the following:

1. Enable mixed precision training:
```python
training_args = SpeechTrainingArguments(
    fp16=True,
    # ...
)
```

2. Use a smaller dataset for testing:
```python
dataloaders = create_librispeech_dataloaders(
    # ...
    max_samples={"train.100": 100, "validation.clean": 10}
)
```

#### Dataset Loading Issues

If you encounter issues loading the dataset, try the following:

1. Use streaming mode:
```python
dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
```

2. Download a smaller subset:
```python
dataset = load_dataset("librispeech_asr", "clean", split="train.100[:100]")
```

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the documentation in the `docs/` directory
2. Look for error messages in the Colab output
3. Consult the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
4. Consult the [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
5. Open an issue on the GitHub repository
