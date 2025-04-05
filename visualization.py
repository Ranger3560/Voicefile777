"""
Visualization utilities for speech recognition models.

This module contains functions for visualizing model performance, training progress,
and audio data for speech recognition tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import pandas as pd
import seaborn as sns
from IPython.display import Audio, display


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Audio Waveform",
    figsize: Tuple[int, int] = (10, 3),
    save_path: Optional[str] = None
):
    """
    Plot audio waveform.
    
    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
):
    """
    Plot spectrogram of audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Plot spectrogram
    librosa.display.specshow(
        D,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
):
    """
    Plot mel spectrogram of audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot mel spectrogram
    librosa.display.specshow(
        log_mel_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def play_audio(audio: np.ndarray, sr: int = 16000):
    """
    Play audio in notebook.
    
    Args:
        audio: Audio array
        sr: Sample rate
    """
    display(Audio(audio, rate=sr))


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Dictionary of training metrics
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Plot each metric
    for i, (metric, values) in enumerate(history.items()):
        plt.subplot(len(history), 1, i+1)
        plt.plot(values)
        plt.title(f'{metric}')
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_wer_distribution(
    wer_values: List[float],
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "WER Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of Word Error Rate values.
    
    Args:
        wer_values: List of WER values
        bins: Number of histogram bins
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(wer_values, bins=bins, alpha=0.7)
    
    # Add mean and median lines
    mean_wer = np.mean(wer_values)
    median_wer = np.median(wer_values)
    
    plt.axvline(mean_wer, color='r', linestyle='--', label=f'Mean: {mean_wer:.4f}')
    plt.axvline(median_wer, color='g', linestyle='--', label=f'Median: {median_wer:.4f}')
    
    plt.title(title)
    plt.xlabel('Word Error Rate')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    references: List[str],
    predictions: List[str],
    top_n_words: int = 20,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Word Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for most common words.
    
    Args:
        references: List of reference texts
        predictions: List of predicted texts
        top_n_words: Number of top words to include
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Tokenize references and predictions
    ref_words = [word for text in references for word in text.lower().split()]
    pred_words = [word for text in predictions for word in text.lower().split()]
    
    # Get most common words
    word_counts = {}
    for word in ref_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
    top_words = [word for word, _ in top_words]
    
    # Create confusion matrix
    confusion = {}
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()
        
        # Align tokens (simple approach)
        for i in range(min(len(ref_tokens), len(pred_tokens))):
            ref_word = ref_tokens[i]
            pred_word = pred_tokens[i]
            
            if ref_word in top_words:
                if ref_word not in confusion:
                    confusion[ref_word] = {}
                
                confusion[ref_word][pred_word] = confusion[ref_word].get(pred_word, 0) + 1
    
    # Convert to DataFrame
    matrix_data = []
    for ref_word, preds in confusion.items():
        for pred_word, count in preds.items():
            matrix_data.append({
                'Reference': ref_word,
                'Prediction': pred_word,
                'Count': count
            })
    
    df = pd.DataFrame(matrix_data)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='Reference',
        columns='Prediction',
        values='Count',
        fill_value=0
    )
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    input_tokens: List[str],
    output_tokens: List[str],
    layer: int = 0,
    head: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Plot attention weights.
    
    Args:
        attention_weights: Attention weights tensor of shape [layers, heads, tgt_len, src_len]
        input_tokens: Input tokens
        output_tokens: Output tokens
        layer: Layer index
        head: Head index
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Extract weights for specified layer and head
    weights = attention_weights[layer, head]
    
    # Plot heatmap
    sns.heatmap(
        weights,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap='viridis'
    )
    
    plt.title(f"{title} (Layer {layer}, Head {head})")
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_error_analysis_report(
    references: List[str],
    predictions: List[str],
    audio_paths: Optional[List[str]] = None,
    output_dir: str = "./error_analysis",
    top_n: int = 10
):
    """
    Create comprehensive error analysis report.
    
    Args:
        references: List of reference texts
        predictions: List of predicted texts
        audio_paths: List of paths to audio files (optional)
        output_dir: Output directory for report
        top_n: Number of top examples to include
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute WER for each example
    wer_values = []
    for ref, pred in zip(references, predictions):
        wer = calculate_wer([ref], [pred])
        wer_values.append(wer)
    
    # Create DataFrame
    data = {
        'Reference': references,
        'Prediction': predictions,
        'WER': wer_values
    }
    
    if audio_paths:
        data['Audio Path'] = audio_paths
    
    df = pd.DataFrame(data)
    
    # Sort by WER
    df = df.sort_values('WER', ascending=False)
    
    # Save full report
    df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
    
    # Plot WER distribution
    plot_wer_distribution(
        wer_values,
        title="Word Error Rate Distribution",
        save_path=os.path.join(output_dir, 'wer_distribution.png')
    )
    
    # Get worst examples
    worst_examples = df.head(top_n)
    
    # Create report for worst examples
    with open(os.path.join(output_dir, 'worst_examples.txt'), 'w') as f:
        f.write("Top Errors by WER\n")
        f.write("=================\n\n")
        
        for i, row in worst_examples.iterrows():
            f.write(f"Example {i+1} (WER: {row['WER']:.4f})\n")
            f.write(f"Reference: {row['Reference']}\n")
            f.write(f"Prediction: {row['Prediction']}\n")
            
            if 'Audio Path' in row:
                f.write(f"Audio: {row['Audio Path']}\n")
            
            f.write("\n")
    
    # Analyze common error patterns
    error_patterns = []
    for ref, pred in zip(references, predictions):
        ref_words = ref.lower().split()
        pred_words = pred.lower().split()
        
        # Simple word-level alignment
        for i in range(min(len(ref_words), len(pred_words))):
            if ref_words[i] != pred_words[i]:
                error_patterns.append((ref_words[i], pred_words[i]))
    
    # Count error patterns
    pattern_counts = {}
    for ref, pred in error_patterns:
        key = f"{ref} → {pred}"
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
    
    # Sort by frequency
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Save error patterns
    with open(os.path.join(output_dir, 'error_patterns.txt'), 'w') as f:
        f.write("Common Error Patterns\n")
        f.write("====================\n\n")
        
        for pattern, count in sorted_patterns[:top_n]:
            f.write(f"{pattern}: {count} occurrences\n")
    
    print(f"Error analysis report saved to {output_dir}")
