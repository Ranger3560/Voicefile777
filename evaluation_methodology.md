# Evaluation Methodology

This document provides a detailed overview of the evaluation methodology for the speech recognition model. It covers metrics, evaluation pipeline, error analysis, and visualization tools.

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Evaluation Pipeline](#evaluation-pipeline)
4. [Error Analysis](#error-analysis)
5. [Visualization Tools](#visualization-tools)
6. [Best Practices](#best-practices)

## Overview

Evaluating speech recognition models requires specialized metrics and analysis techniques to understand model performance. Our evaluation methodology is designed to provide comprehensive insights into model performance, error patterns, and areas for improvement.

The evaluation process is implemented in the `scripts/evaluate.py` and `utils/metrics.py` files, which provide a command-line interface and a robust evaluation framework.

## Evaluation Metrics

### Word Error Rate (WER)

Word Error Rate (WER) is the primary metric for evaluating speech recognition models. It measures the minimum number of word edits (insertions, deletions, and substitutions) required to transform the predicted text into the reference text, divided by the number of words in the reference:

```
WER = (S + D + I) / N
```

Where:
- S: Number of substitutions
- D: Number of deletions
- I: Number of insertions
- N: Number of words in the reference

```python
def compute_wer(predictions, references):
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute WER
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
```

### Character Error Rate (CER)

Character Error Rate (CER) is similar to WER but operates at the character level. It measures the minimum number of character edits required to transform the predicted text into the reference text, divided by the number of characters in the reference:

```
CER = (S + D + I) / N
```

Where:
- S: Number of character substitutions
- D: Number of character deletions
- I: Number of character insertions
- N: Number of characters in the reference

```python
def compute_cer(predictions, references):
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute CER
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
```

### Word Accuracy

Word Accuracy is the complement of WER:

```
Word Accuracy = 1 - WER
```

```python
def compute_word_accuracy(predictions, references):
    wer = compute_wer(predictions, references)
    return 1.0 - wer
```

### Sentence Accuracy

Sentence Accuracy measures the proportion of sentences that are perfectly transcribed:

```python
def compute_sentence_accuracy(predictions, references):
    # Normalize text
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    
    # Compute sentence accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)
```

## Evaluation Pipeline

### Command-Line Interface

The evaluation pipeline is accessible through a command-line interface:

```bash
python scripts/evaluate.py \
    --model_path="./output/final_model" \
    --dataset_name="librispeech_asr" \
    --dataset_config="clean" \
    --eval_split="test.clean" \
    --batch_size=16 \
    --output_dir="./evaluation_results" \
    --save_predictions
```

### Evaluation Process

The evaluation process consists of the following steps:

1. **Load Model and Processor**: Load the trained model and processor
2. **Load Evaluation Dataset**: Load the evaluation dataset
3. **Generate Predictions**: Generate predictions for the evaluation dataset
4. **Compute Metrics**: Compute evaluation metrics
5. **Analyze Errors**: Analyze error patterns
6. **Generate Visualizations**: Generate visualizations of the results
7. **Create Report**: Create an evaluation report

```python
def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    model, processor = create_whisper_model_and_processor(
        model_name_or_path=args.model_path,
        use_enhanced_model=args.use_enhanced_model
    )
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataloaders = create_librispeech_dataloaders(
        train_split=None,
        eval_split=args.eval_split,
        config=args.dataset_config,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        max_samples=max_samples
    )
    
    eval_dataloader = dataloaders["eval"]
    
    # Evaluate model
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
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
        output_dir=args.output_dir
    )
    
    # Create evaluation report
    report_path = create_evaluation_report(
        metrics,
        output_file=os.path.join(args.output_dir, "evaluation_report.md")
    )
```

## Error Analysis

### Detailed WER Metrics

We compute detailed WER metrics to understand the types of errors the model makes:

```python
def compute_detailed_wer_metrics(predictions, references):
    # Compute detailed WER metrics
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
```

### WER by Sentence Length

We analyze how WER varies with sentence length to identify potential issues with long or short sentences:

```python
def compute_word_error_by_length(predictions, references):
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
```

### Word Confusion Matrix

We compute a word confusion matrix to identify specific words that the model struggles with:

```python
def compute_confusion_matrix(predictions, references, top_k=10):
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
```

## Visualization Tools

### WER by Sentence Length

We visualize how WER varies with sentence length:

```python
def visualize_wer_by_length(wer_by_length, output_file=None):
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
```

### Detailed WER Metrics

We visualize the detailed WER metrics:

```python
def visualize_detailed_wer_metrics(metrics, output_file=None):
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
```

### Word Confusion Matrix

We visualize the word confusion matrix:

```python
def visualize_confusion_matrix(confusion_matrix, output_file=None, top_n=10):
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
```

### Example Predictions

We create an HTML file with example predictions to manually inspect model outputs:

```python
def save_example_predictions(predictions, references, output_file):
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Example Predictions</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .correct {
                color: green;
            }
            .incorrect {
                color: red;
            }
        </style>
    </head>
    <body>
        <h1>Example Predictions</h1>
        <table>
            <tr>
                <th>#</th>
                <th>Reference</th>
                <th>Prediction</th>
                <th>WER</th>
            </tr>
    """
    
    # Add examples
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Compute WER for this example
        example_wer = compute_wer([pred], [ref])
        
        # Determine class based on WER
        if example_wer == 0:
            pred_class = "correct"
        else:
            pred_class = "incorrect"
        
        # Add row
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{ref}</td>
                <td class="{pred_class}">{pred}</td>
                <td>{example_wer:.3f}</td>
            </tr>
        """
    
    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(html_content)
    
    return output_file
```

## Best Practices

### Evaluation Dataset Selection

For comprehensive evaluation, we recommend using multiple test sets:

- **Clean Speech**: Test set with clean, studio-quality speech
- **Noisy Speech**: Test set with background noise
- **Accented Speech**: Test set with various accents
- **Domain-Specific**: Test set from the target domain

### Normalization

Before computing metrics, normalize both predictions and references:

- Convert to lowercase
- Remove punctuation (except apostrophes)
- Remove extra whitespace
- Strip leading and trailing whitespace

```python
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    punctuation = string.punctuation.
(Content truncated due to size limit. Use line ranges to read in chunks)