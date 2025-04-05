"""
Evaluation metrics for speech recognition models.

This module contains functions for computing evaluation metrics for speech recognition models,
including Word Error Rate (WER), Character Error Rate (CER), and other metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
import re
import string
import logging
import jiwer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Word Error Rate
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for WER calculation")
        return 1.0
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute WER
    try:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(),
            jiwer.RemoveEmptyStrings()
        ])
        wer = jiwer.wer(
            references, 
            predictions,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
        return wer
    except Exception as e:
        logger.error(f"Error computing WER: {e}")
        return 1.0


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Character Error Rate
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for CER calculation")
        return 1.0
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute CER
    try:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
            jiwer.RemoveEmptyStrings()
        ])
        cer = jiwer.wer(
            references, 
            predictions,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
        return cer
    except Exception as e:
        logger.error(f"Error computing CER: {e}")
        return 1.0


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punctuation))
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    return text


def compute_word_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute word accuracy (1 - WER).
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Word accuracy
    """
    wer = compute_wer(predictions, references)
    return 1.0 - wer


def compute_sentence_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute sentence accuracy (exact match).
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Sentence accuracy
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for sentence accuracy calculation")
        return 0.0
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute sentence accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def compute_detailed_wer_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute detailed WER metrics including insertions, deletions, and substitutions.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Dictionary of detailed metrics
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for detailed WER calculation")
        return {
            "wer": 1.0,
            "insertions": 0.0,
            "deletions": 0.0,
            "substitutions": 0.0
        }
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute detailed WER metrics
    try:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(),
            jiwer.RemoveEmptyStrings()
        ])
        
        measures = jiwer.compute_measures(
            references, 
            predictions,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
        
        return {
            "wer": measures["wer"],
            "insertions": measures["insertions"] / measures["words"],
            "deletions": measures["deletions"] / measures["words"],
            "substitutions": measures["substitutions"] / measures["words"]
        }
    except Exception as e:
        logger.error(f"Error computing detailed WER metrics: {e}")
        return {
            "wer": 1.0,
            "insertions": 0.0,
            "deletions": 0.0,
            "substitutions": 0.0
        }


def compute_word_error_by_length(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute WER grouped by reference sentence length.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Dictionary of WER by sentence length
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for WER by length calculation")
        return {}
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Group by sentence length
    length_groups = defaultdict(list)
    for p, r in zip(predictions, references):
        r_words = r.split()
        length = len(r_words)
        
        # Group by length ranges
        if length <= 5:
            length_group = "1-5"
        elif length <= 10:
            length_group = "6-10"
        elif length <= 15:
            length_group = "11-15"
        elif length <= 20:
            length_group = "16-20"
        else:
            length_group = "21+"
        
        length_groups[length_group].append((p, r))
    
    # Compute WER for each length group
    wer_by_length = {}
    for length_group, pairs in length_groups.items():
        group_predictions = [p for p, _ in pairs]
        group_references = [r for _, r in pairs]
        wer_by_length[length_group] = compute_wer(group_predictions, group_references)
    
    return wer_by_length


def compute_confusion_matrix(predictions: List[str], references: List[str], top_k: int = 10) -> Dict[str, Dict[str, int]]:
    """
    Compute word confusion matrix.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        top_k: Number of top confusions to return
        
    Returns:
        Dictionary of word confusions
    """
    # Ensure inputs are valid
    if not predictions or not references:
        logger.warning("Empty inputs for confusion matrix calculation")
        return {}
    
    if len(predictions) != len(references):
        logger.warning(f"Mismatched lengths: predictions={len(predictions)}, references={len(references)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute alignments
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.SentencesToListOfWords(),
        jiwer.RemoveEmptyStrings()
    ])
    
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for p, r in zip(predictions, references):
        # Transform sentences to word lists
        p_words = transformation(p)
        r_words = transformation(r)
        
        # Compute alignment
        alignment = jiwer.process_words(r_words, p_words)
        
        # Extract substitutions
        for r_word, p_word in alignment.substitutions:
            confusion_matrix[r_word][p_word] += 1
    
    # Convert to regular dictionary and keep only top_k confusions per word
    result = {}
    for r_word, confusions in confusion_matrix.items():
        # Sort confusions by count in descending order
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only top_k confusions
        result[r_word] = {p_word: count for p_word, count in sorted_confusions[:top_k]}
    
    return result


def visualize_wer_by_length(wer_by_length: Dict[str, float], output_file: Optional[str] = None) -> str:
    """
    Visualize WER by sentence length.
    
    Args:
        wer_by_length: Dictionary of WER by sentence length
        output_file: Output file path
        
    Returns:
        Output file path
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Sort length groups
    length_order = ["1-5", "6-10", "11-15", "16-20", "21+"]
    lengths = []
    wers = []
    
    for length in length_order:
        if length in wer_by_length:
            lengths.append(length)
            wers.append(wer_by_length[length])
    
    # Create bar chart
    plt.bar(lengths, wers)
    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Word Error Rate")
    plt.title("WER by Sentence Length")
    plt.ylim(0, min(1.0, max(wers) * 1.2))  # Set y-axis limit
    
    # Add values on top of bars
    for i, wer in enumerate(wers):
        plt.text(i, wer + 0.01, f"{wer:.3f}", ha="center")
    
    # Save figure
    if output_file is None:
        output_file = "wer_by_length.png"
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file


def visualize_confusion_matrix(confusion_matrix: Dict[str, Dict[str, int]], output_file: Optional[str] = None, top_n: int = 10) -> str:
    """
    Visualize word confusion matrix.
    
    Args:
        confusion_matrix: Dictionary of word confusions
        output_file: Output file path
        top_n: Number of top confusions to visualize
        
    Returns:
        Output file path
    """
    # Get top confusions
    all_confusions = []
    for r_word, confusions in confusion_matrix.items():
        for p_word, count in confusions.items():
            all_confusions.append((r_word, p_word, count))
    
    # Sort by count in descending order
    all_confusions.sort(key=lambda x: x[2], reverse=True)
    
    # Keep only top_n confusions
    top_confusions = all_confusions[:top_n]
    
    # Create dataframe
    df = pd.DataFrame(top_confusions, columns=["Reference", "Prediction", "Count"])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    pivot_df = df.pivot(index="Reference", columns="Prediction", values="Count")
    sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlGnBu")
    
    plt.title(f"Top {top_n} Word Confusions")
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = "confusion_matrix.png"
    
    plt.savefig(output_file)
    plt.close()
    
    return output_file


def visualize_detailed_wer_metrics(metrics: Dict[str, float], output_file: Optional[str] = None) -> str:
    """
    Visualize detailed WER metrics.
    
    Args:
        metrics: Dictionary of detailed WER metrics
        output_file: Output file path
        
    Returns:
        Output file path
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Extract metrics
    labels = ["Insertions", "Deletions", "Substitutions"]
    values = [metrics["insertions"], metrics["deletions"], metrics["substitutions"]]
    
    # Create bar chart
    plt.bar(labels, values)
    plt.xlabel("Error Type")
    plt.ylabel("Rate")
    plt.title(f"Detailed WER Metrics (Total WER: {metrics['wer']:.3f})")
    plt.ylim(0, min(1.0, max(values) * 1.2))  # Set y-axis limit
    
    # Add values on top of bars
    for i, value in enumerate(values):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center")
    
    # Save figure
    if output_file is None:
        output_file = "detailed_wer_metrics.png"
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file


def evaluate_model(
    predictions: List[str],
    references: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance and generate visualizations.
    
    Args:
        predictions: List of predicted transcriptions
        refere
(Content truncated due to size limit. Use line ranges to read in chunks)