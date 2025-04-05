# API Documentation

This document provides a detailed overview of the API for the speech recognition model. It covers model loading, inference, configuration, and utility functions.

## Table of Contents

1. [Overview](#overview)
2. [Model Loading](#model-loading)
3. [Inference API](#inference-api)
4. [Configuration API](#configuration-api)
5. [Utility Functions](#utility-functions)
6. [Examples](#examples)

## Overview

The speech recognition model API provides a simple interface for loading models, performing inference, and configuring model parameters. The API is designed to be easy to use while providing access to advanced features when needed.

## Model Loading

### Loading Pre-trained Models

```python
from models.whisper_model import create_whisper_model_and_processor

# Load base model
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny"
)

# Load enhanced model
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",
    use_enhanced_model=True
)

# Load from local directory
model, processor = create_whisper_model_and_processor(
    model_name_or_path="./output/final_model"
)
```

### Loading with Custom Configuration

```python
from models.config import WhisperModelConfig
from models.whisper_model import create_whisper_model_and_processor

# Create custom configuration
config = WhisperModelConfig(
    model_size="tiny",
    use_multilingual=False,
    num_encoder_layers=6,
    num_decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    d_model=512
)

# Load model with custom configuration
model, processor = create_whisper_model_and_processor(
    model_name_or_path=None,
    custom_config=config
)
```

## Inference API

### Basic Transcription

```python
import torch
import librosa

# Load audio file
audio, sr = librosa.load("audio.wav", sr=16000)

# Convert to tensor
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    output = model.generate(input_features)

# Decode output
transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
print(transcription)
```

### Batch Transcription

```python
import torch
import librosa
import numpy as np

# Load multiple audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
audio_tensors = []

for file in audio_files:
    audio, sr = librosa.load(file, sr=16000)
    audio_tensors.append(audio)

# Process batch
input_features = processor(
    audio_tensors, 
    sampling_rate=16000, 
    return_tensors="pt", 
    padding=True
).input_features

# Generate transcriptions
with torch.no_grad():
    output = model.generate(input_features)

# Decode output
transcriptions = processor.batch_decode(output, skip_special_tokens=True)
for i, transcription in enumerate(transcriptions):
    print(f"Audio {i+1}: {transcription}")
```

### Advanced Transcription Options

```python
import torch
import librosa

# Load audio file
audio, sr = librosa.load("audio.wav", sr=16000)

# Convert to tensor
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

# Generate transcription with advanced options
with torch.no_grad():
    output = model.generate(
        input_features,
        max_length=256,
        num_beams=5,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

# Decode output
transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
print(transcription)
```

### Streaming Transcription

```python
import torch
import numpy as np
import sounddevice as sd

# Define callback function for streaming
def callback(indata, frames, time, status):
    # Process audio chunk
    audio_chunk = indata[:, 0]  # Mono audio
    
    # Convert to tensor
    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    
    # Generate transcription
    with torch.no_grad():
        output = model.generate(input_features)
    
    # Decode output
    transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(transcription)

# Start streaming
with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=16000):
    print("Listening... Press Ctrl+C to stop")
    while True:
        sd.sleep(1000)  # Sleep for 1 second
```

## Configuration API

### Model Configuration

```python
from models.config import WhisperModelConfig

# Create configuration
config = WhisperModelConfig(
    model_size="tiny",  # tiny, base, small, medium, large
    use_multilingual=False,
    num_encoder_layers=6,
    num_decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    d_model=512,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048
)

# Save configuration
config.save_to_json("config.json")

# Load configuration
config = WhisperModelConfig.from_json("config.json")
```

### Training Configuration

```python
from scripts.trainer import SpeechTrainingArguments

# Create training arguments
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

# Save arguments
training_args.save_to_json("training_args.json")

# Load arguments
training_args = SpeechTrainingArguments.from_json("training_args.json")
```

## Utility Functions

### Audio Preprocessing

```python
from data.preprocessing import preprocess_audio, extract_features

# Preprocess audio
audio_processed = preprocess_audio(
    audio,
    sample_rate=16000,
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
```

### Data Augmentation

```python
from data.augmentation import augment_audio

# Apply augmentation
augmented_audio = augment_audio(
    audio,
    sample_rate=16000,
    time_stretch_factor=0.9,
    pitch_shift_steps=2,
    add_noise=True,
    noise_level=0.005
)
```

### Metrics Calculation

```python
from utils.metrics import compute_wer, compute_cer

# Calculate WER
wer = compute_wer(predictions, references)

# Calculate CER
cer = compute_cer(predictions, references)

# Calculate detailed metrics
detailed_metrics = compute_detailed_wer_metrics(predictions, references)
```

### Model Optimization

```python
from utils.optimization import apply_dynamic_quantization, apply_pruning, export_to_onnx

# Apply quantization
quantized_model = apply_dynamic_quantization(model)

# Apply pruning
pruned_model = apply_pruning(model, pruning_method="l1_unstructured", amount=0.2)

# Export to ONNX
onnx_path = export_to_onnx(
    model,
    output_path="model.onnx",
    input_shape=(1, 80, 3000)
)
```

## Examples

### Complete Transcription Example

```python
import torch
import librosa
import numpy as np
from models.whisper_model import create_whisper_model_and_processor
from data.preprocessing import preprocess_audio, extract_features

# Load model and processor
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",
    use_enhanced_model=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load and preprocess audio
audio, sr = librosa.load("audio.wav", sr=16000)
audio_processed = preprocess_audio(
    audio,
    sample_rate=16000,
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
transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

### Training Example

```python
import torch
from models.whisper_model import create_whisper_model_and_processor
from data.librispeech import create_librispeech_dataloaders, create_data_collator
from scripts.trainer import SpeechTrainer, SpeechTrainingArguments

# Load model and processor
model, processor = create_whisper_model_and_processor(
    model_name_or_path="openai/whisper-tiny",
    use_enhanced_model=True
)

# Create data collator
data_collator = create_data_collator(processor)

# Create dataloaders
dataloaders = create_librispeech_dataloaders(
    train_split="train.100",
    eval_split="validation.clean",
    config="clean",
    batch_size=8,
    collate_fn=data_collator
)

train_dataloader = dataloaders["train"]
eval_dataloader = dataloaders["eval"]

# Create training arguments
training_args = SpeechTrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    early_stopping_patience=3
)

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

# Save final model
model.save_pretrained("./output/final_model")
processor.save_pretrained("./output/final_model")
```

### Evaluation Example

```python
import torch
from models.whisper_model import create_whisper_model_and_processor
from data.librispeech import create_librispeech_dataloaders, create_data_collator
from utils.metrics import evaluate_model, create_evaluation_report

# Load model and processor
model, processor = create_whisper_model_and_processor(
    model_name_or_path="./output/final_model"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Create data collator
data_collator = create_data_collator(processor)

# Create dataloader
dataloaders = create_librispeech_dataloaders(
    train_split=None,
    eval_split="test.clean",
    config="clean",
    batch_size=16,
    collate_fn=data_collator
)

eval_dataloader = dataloaders["eval"]

# Evaluate model
all_predictions = []
all_references = []

with torch.no_grad():
    for batch in eval_dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Generate predictions
        outputs = model(**batch)
        pred_ids = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        label_ids = batch["labels"].detach().cpu().numpy()
        
        # Decode predictions and references
        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Store predictions and references
        all_predictions.extend(predictions)
        all_references.extend(references)

# Evaluate model
metrics = evaluate_model(
    all_predictions,
    all_references,
    output_dir="./evaluation_results"
)

# Create evaluation report
report_path = create_evaluation_report(
    metrics,
    output_file="./evaluation_results/evaluation_report.md"
)

# Print summary metrics
print(f"Word Error Rate (WER): {metrics['wer']:.4f}")
print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
print(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
print(f"Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
```

### Optimization Example

```python
import torch
from models.whisper_model import create_whisper_model_and_processor
from utils.optimization import (
    apply_dynamic_quantization,
    apply_pruning,
    export_to_onnx,
    optimize_onnx_model,
    create_tensorrt_engine,
    benchmark_model,
    benchmark_onnx_model,
    benchmark_tensorrt_engine
)

# Load model
model, processor = create_whisper_model_and_processor(
    model_name_or_path="./output/final_model"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define input shape
input_shape = (1, 80, 3000)

# Benchmark original model
original_benchmark = benchmark_model(
    model=model,
    input_shape=input_shape,
    device=device,
    num_iterations=100,
    warmup_iterations=10
)

# Apply quantization
quantized_model = apply_dynamic_quantization(model)

# Benchmark quantized model
quantized_benchmark = benchmark_model(
    model=quantized_model,
    input_shape=input_shape,
    device=device,
    num_iterations=100,
    warmup_iterations=10
)

# Apply pruning
pruned_model = apply_pruning(
    model=model,
    pruning_method="l1_unstructured",
    amount=0.2
)

# Benchmark pruned model
pruned_benchmark = benchmark_model(
    model=pruned_model,
    input_shape=input_shape,
    device=device,
    num_iterations=100,
    warmup_iterations=10
)

# Export to ONNX
onnx_path = export_to_onnx(
    model=model,
    output_path="./optimized_models/model.onnx",
    input_shape=input_shape
)

# Optimize ONNX model
optimized_onnx_path = optimize_onnx_model(
    input_path=onnx_path,
    output_path="./optimized_models/model_optimized.onnx"
)

# Benchmark ONNX model
onnx_benchmark = benchmark_onnx_model(
    onnx_path=optimized_onnx_path,
    input_shape=input_shape,
    num_iterations=100,
    warmup_iterations=10
)

# Create TensorRT engine
tensorrt_path = create_tensorrt_engine(
    onnx_path=optimized_onnx_path,
    engine_path="./optimized_models/model_fp16.engine",
    precision="fp16"
)

# Benchmark TensorRT engine
tensorrt_benchmark = benchmark_tensorrt_engine(
    engine_path=tensorrt_path,
    input_shape=input_shape,
    num_iterations=100,
    warmup_iterations=10
)

# Print benchmark results
print("Benchmark Results:")
print(f"Original Model: {original_benchmark['latency_ms']:.2f} ms")
print(f"Quantized Model: {quantized_benchmark['latency_ms']:.2f} ms")
print(f"Pruned Model: {pruned_benchmark['latency_ms']:.2f} ms")
print(f"ONNX Model: {onnx_benchmark['latency_ms']:.2f} ms")
print(f"TensorRT Engine: {tensorrt_benchmark['latency_ms']:.2f} ms")
```
