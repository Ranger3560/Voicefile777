"""
Evaluation script for speech recognition models.

This script provides a command-line interface for evaluating speech recognition models
using the metrics module.
"""

import argparse
import os
import sys
import torch
import logging
import json
from tqdm import tqdm
from typing import List, Dict, Any

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.whisper_model import create_whisper_model_and_processor
from data.librispeech import create_librispeech_dataloaders, create_data_collator
from utils.metrics import evaluate_model, create_evaluation_report

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
    parser = argparse.ArgumentParser(description="Evaluate a speech recognition model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--use_enhanced_model",
        action="store_true",
        help="Whether to use the enhanced model with custom components"
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
        "--eval_split",
        type=str,
        default="test.clean",
        help="Evaluation split"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save all predictions"
    )
    
    # Miscellaneous arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
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
    Main function for evaluating a speech recognition model.
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "eval_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_path}")
    model, processor = create_whisper_model_and_processor(
        model_name_or_path=args.model_path,
        use_enhanced_model=args.use_enhanced_model
    )
    model = model.to(device)
    model.eval()
    
    # Create data collator
    data_collator = create_data_collator(processor)
    
    # Create dataloader
    logger.info(f"Loading evaluation dataset: {args.dataset_name} ({args.dataset_config}, {args.eval_split})")
    max_samples = {}
    if args.max_eval_samples is not None:
        max_samples[args.eval_split] = args.max_eval_samples
    
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
    logger.info("Evaluating model")
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate predictions
            if "labels" in batch:
                # Get predictions
                outputs = model(**batch)
                pred_ids = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
                label_ids = batch["labels"].detach().cpu().numpy()
                
                # Replace -100 with pad token ID
                label_ids = [[id if id != -100 else processor.tokenizer.pad_token_id for id in ids] for ids in label_ids]
                
                # Decode predictions and references
                predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
                references = processor.batch_decode(label_ids, skip_special_tokens=True)
                
                # Store predictions and references
                all_predictions.extend(predictions)
                all_references.extend(references)
    
    # Evaluate model
    logger.info("Computing evaluation metrics")
    metrics = evaluate_model(
        all_predictions,
        all_references,
        output_dir=args.output_dir
    )
    
    # Create evaluation report
    logger.info("Creating evaluation report")
    report_path = create_evaluation_report(
        metrics,
        output_file=os.path.join(args.output_dir, "evaluation_report.md")
    )
    
    # Save all predictions if requested
    if args.save_predictions:
        logger.info("Saving all predictions")
        predictions_data = [
            {"reference": ref, "prediction": pred}
            for ref, pred in zip(all_references, all_predictions)
        ]
        
        with open(os.path.join(args.output_dir, "all_predictions.json"), "w") as f:
            json.dump(predictions_data, f, indent=2)
    
    # Print summary metrics
    logger.info("Evaluation completed")
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f}")
    logger.info(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    logger.info(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
    logger.info(f"Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
