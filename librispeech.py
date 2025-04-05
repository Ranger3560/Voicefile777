"""
Data loading pipeline for LibriSpeech dataset.

This module contains functions for loading and processing the LibriSpeech dataset
for speech recognition tasks, including streaming and batch processing options.
"""

import os
import torch
from datasets import load_dataset, Audio, Dataset, DatasetDict
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_librispeech_dataset(
    config: str = "clean",
    split: Union[str, List[str]] = "train.100",
    streaming: bool = False,
    target_sampling_rate: int = 16000
) -> Union[Dataset, DatasetDict]:
    """
    Load LibriSpeech dataset.
    
    Args:
        config: Dataset configuration ("clean", "other", etc.)
        split: Dataset split(s) to load
        streaming: Whether to use streaming mode
        target_sampling_rate: Target sampling rate for audio
        
    Returns:
        LibriSpeech dataset
    """
    logger.info(f"Loading LibriSpeech dataset (config={config}, split={split}, streaming={streaming})")
    
    # Load dataset
    dataset = load_dataset(
        "librispeech_asr",
        config,
        split=split,
        streaming=streaming
    )
    
    # Cast to Audio format
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    
    logger.info(f"LibriSpeech dataset loaded successfully")
    
    return dataset


def create_librispeech_dataloaders(
    train_split: str = "train.100",
    eval_split: str = "validation.clean",
    test_split: Optional[str] = "test.clean",
    config: str = "clean",
    batch_size: int = 8,
    target_sampling_rate: int = 16000,
    preprocess_fn: Optional[Callable] = None,
    filter_fn: Optional[Callable] = None,
    augment_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    max_samples: Optional[Dict[str, int]] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for LibriSpeech dataset.
    
    Args:
        train_split: Training split
        eval_split: Evaluation split
        test_split: Test split (optional)
        config: Dataset configuration
        batch_size: Batch size
        target_sampling_rate: Target sampling rate for audio
        preprocess_fn: Preprocessing function
        filter_fn: Filtering function
        augment_fn: Augmentation function
        collate_fn: Collate function
        num_workers: Number of workers for DataLoader
        max_samples: Maximum number of samples per split
        
    Returns:
        Dictionary of DataLoaders
    """
    logger.info(f"Creating LibriSpeech DataLoaders")
    
    # Define splits to load
    splits = [train_split, eval_split]
    if test_split:
        splits.append(test_split)
    
    # Load datasets
    datasets = {}
    for split in splits:
        dataset = load_librispeech_dataset(
            config=config,
            split=split,
            streaming=False,
            target_sampling_rate=target_sampling_rate
        )
        
        # Apply preprocessing if provided
        if preprocess_fn:
            dataset = dataset.map(preprocess_fn)
        
        # Apply filtering if provided
        if filter_fn:
            dataset = dataset.filter(filter_fn)
        
        # Apply augmentation to training set only
        if augment_fn and split == train_split:
            dataset = dataset.map(augment_fn)
        
        # Limit number of samples if specified
        if max_samples and split in max_samples:
            dataset = dataset.select(range(min(max_samples[split], len(dataset))))
        
        datasets[split] = dataset
    
    # Create DataLoaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == train_split)  # Shuffle only training set
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    # Rename keys to standard names
    key_mapping = {
        train_split: "train",
        eval_split: "eval"
    }
    if test_split:
        key_mapping[test_split] = "test"
    
    renamed_dataloaders = {key_mapping.get(k, k): v for k, v in dataloaders.items()}
    
    logger.info(f"Created DataLoaders with {len(renamed_dataloaders)} splits")
    for split, dataloader in renamed_dataloaders.items():
        logger.info(f"  {split}: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
    
    return renamed_dataloaders


def create_streaming_dataset(
    dataset_name: str = "librispeech_asr",
    config: str = "clean",
    split: str = "train.100",
    target_sampling_rate: int = 16000,
    preprocess_fn: Optional[Callable] = None,
    filter_fn: Optional[Callable] = None,
    buffer_size: int = 1000
) -> Dataset:
    """
    Create a streaming dataset for efficient processing of large datasets.
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        split: Dataset split
        target_sampling_rate: Target sampling rate for audio
        preprocess_fn: Preprocessing function
        filter_fn: Filtering function
        buffer_size: Buffer size for shuffling
        
    Returns:
        Streaming dataset
    """
    logger.info(f"Creating streaming dataset for {dataset_name} (config={config}, split={split})")
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True
    )
    
    # Cast to Audio format
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    
    # Apply preprocessing if provided
    if preprocess_fn:
        dataset = dataset.map(preprocess_fn)
    
    # Apply filtering if provided
    if filter_fn:
        dataset = dataset.filter(filter_fn)
    
    # Shuffle with buffer
    dataset = dataset.shuffle(buffer_size=buffer_size)
    
    logger.info(f"Streaming dataset created successfully")
    
    return dataset


def create_custom_dataset_from_directory(
    audio_dir: str,
    transcript_file: str,
    target_sampling_rate: int = 16000,
    preprocess_fn: Optional[Callable] = None,
    filter_fn: Optional[Callable] = None,
    audio_format: str = "wav"
) -> Dataset:
    """
    Create a custom dataset from a directory of audio files and a transcript file.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: Path to transcript file (format: "file_id transcript")
        target_sampling_rate: Target sampling rate for audio
        preprocess_fn: Preprocessing function
        filter_fn: Filtering function
        audio_format: Audio file format
        
    Returns:
        Custom dataset
    """
    import pandas as pd
    import librosa
    
    logger.info(f"Creating custom dataset from {audio_dir}")
    
    # Load transcripts
    transcripts = pd.read_csv(transcript_file, sep="\t", header=None, names=["file_id", "text"])
    
    # Create dataset
    data = []
    for _, row in transcripts.iterrows():
        file_id = row["file_id"]
        text = row["text"]
        
        # Find audio file
        audio_path = os.path.join(audio_dir, f"{file_id}.{audio_format}")
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=target_sampling_rate)
            
            # Create sample
            sample = {
                "file_id": file_id,
                "audio": {"array": audio, "sampling_rate": sr},
                "text": text
            }
            
            # Apply preprocessing if provided
            if preprocess_fn:
                sample = preprocess_fn(sample)
            
            # Apply filtering if provided
            if filter_fn and not filter_fn(sample):
                continue
            
            data.append(sample)
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    logger.info(f"Created custom dataset with {len(dataset)} samples")
    
    return dataset


def create_data_collator(processor):
    """
    Create a data collator function for the Whisper model.
    
    Args:
        processor: Whisper processor
        
    Returns:
        Data collator function
    """
    def collator(features):
        # Extract audio and text
        input_features = [feature["audio"] for feature in features]
        label_features = [feature["text"] for feature in features]
        
        # Process audio inputs
        input_features = processor.feature_extractor(
            input_features,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Process text labels
        with processor.as_target_processor():
            labels = processor.tokenizer(
                label_features,
                return_tensors="pt",
                padding=True
            )
        
        # Replace padding token id with -100 for loss calculation
        labels["input_ids"][labels["input_ids"] == processor.tokenizer.pad_token_id] = -100
        
        # Combine inputs and labels
        batch = {
            "input_features": input_features.input_features,
            "labels": labels["input_ids"]
        }
        
        return batch
    
    return collator


def get_dataset_statistics(dataset, num_samples=1000):
    """
    Get statistics about a dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary of statistics
    """
    logger.info(f"Computing dataset statistics (sampling {num_samples} examples)")
    
    # Sample dataset
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        samples = [dataset[i] for i in indices]
    else:
        samples = [dataset[i] for i in range(len(dataset))]
    
    # Compute statistics
    audio_lengths = []
    text_lengths = []
    
    for sample in samples:
        if "audio" in sample and isinstance(sample["audio"], dict) and "array" in sample["audio"]:
            audio_lengths.append(len(sample["audio"]["array"]))
        elif "audio" in sample and torch.is_tensor(sample["audio"]):
            audio_lengths.append(len(sample["audio"]))
        
        if "text" in sample and isinstance(sample["text"], str):
            text_lengths.append(len(sample["text"].split()))
    
    # Compute statistics
    stats = {
        "num_samples": len(dataset),
        "audio_length": {
            "min": min(audio_lengths) if audio_lengths else 0,
            "max": max(audio_lengths) if audio_lengths else 0,
            "mean": np.mean(audio_lengths) if audio_lengths else 0,
            "median": np.median(audio_lengths) if audio_lengths else 0,
            "std": np.std(audio_lengths) if audio_lengths else 0
        },
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
            "mean": np.mean(text_lengths) if text_lengths else 0,
            "median": np.median(text_lengths) if text_lengths else 0,
            "std": np.std(text_lengths) if text_lengths else 0
        }
    }
    
    logger.info(f"Dataset statistics computed successfully")
    
    return stats
