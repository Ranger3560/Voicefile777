"""
Optimization script for speech recognition models.

This script provides a command-line interface for optimizing speech recognition models
using the optimization module.
"""

import argparse
import os
import sys
import torch
import logging
import json
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.whisper_model import create_whisper_model_and_processor
from utils.optimization import (
    apply_dynamic_quantization,
    apply_pruning,
    make_pruning_permanent,
    export_to_onnx,
    optimize_onnx_model,
    create_tensorrt_engine,
    benchmark_model,
    benchmark_onnx_model,
    benchmark_tensorrt_engine,
    save_benchmark_results,
    visualize_benchmark_results
)

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
    parser = argparse.ArgumentParser(description="Optimize a speech recognition model")
    
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
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./optimized_models",
        help="Output directory for optimized models"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--apply_quantization",
        action="store_true",
        help="Whether to apply quantization"
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Type of quantization to apply"
    )
    parser.add_argument(
        "--apply_pruning",
        action="store_true",
        help="Whether to apply pruning"
    )
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="l1_unstructured",
        choices=["l1_unstructured", "random_unstructured"],
        help="Pruning method to apply"
    )
    parser.add_argument(
        "--pruning_amount",
        type=float,
        default=0.2,
        help="Amount of parameters to prune (0.0 to 1.0)"
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Whether to export model to ONNX format"
    )
    parser.add_argument(
        "--optimize_onnx",
        action="store_true",
        help="Whether to optimize ONNX model"
    )
    parser.add_argument(
        "--create_tensorrt",
        action="store_true",
        help="Whether to create TensorRT engine"
    )
    parser.add_argument(
        "--tensorrt_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode for TensorRT engine"
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--run_benchmarks",
        action="store_true",
        help="Whether to run benchmarks"
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1,80,3000",
        help="Input shape for benchmarking (comma-separated)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking"
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations for benchmarking"
    )
    
    # Miscellaneous arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for optimization"
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
    Main function for optimizing a speech recognition model.
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "optimize_args.json"), "w") as f:
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
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))
    
    # Initialize benchmark results
    benchmark_results = {}
    
    # Benchmark original model
    if args.run_benchmarks:
        logger.info("Benchmarking original model")
        benchmark_results["original"] = benchmark_model(
            model=model,
            input_shape=input_shape,
            device=device,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations
        )
    
    # Apply quantization
    if args.apply_quantization:
        logger.info(f"Applying {args.quantization_type} quantization")
        
        if args.quantization_type == "dynamic":
            quantized_model = apply_dynamic_quantization(
                model=model,
                dtype=torch.qint8
            )
        else:
            # TODO: Implement static quantization with calibration data
            logger.warning("Static quantization not fully implemented, using dynamic quantization instead")
            quantized_model = apply_dynamic_quantization(
                model=model,
                dtype=torch.qint8
            )
        
        # Save quantized model
        quantized_model_path = os.path.join(args.output_dir, "quantized_model")
        os.makedirs(quantized_model_path, exist_ok=True)
        
        if hasattr(quantized_model, "save_pretrained"):
            quantized_model.save_pretrained(quantized_model_path)
        else:
            torch.save(quantized_model.state_dict(), os.path.join(quantized_model_path, "model.pt"))
        
        # Save processor
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(quantized_model_path)
        
        logger.info(f"Quantized model saved to {quantized_model_path}")
        
        # Benchmark quantized model
        if args.run_benchmarks:
            logger.info("Benchmarking quantized model")
            benchmark_results["quantized"] = benchmark_model(
                model=quantized_model,
                input_shape=input_shape,
                device=device,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations
            )
    
    # Apply pruning
    if args.apply_pruning:
        logger.info(f"Applying {args.pruning_method} pruning with amount {args.pruning_amount}")
        
        pruned_model = apply_pruning(
            model=model,
            pruning_method=args.pruning_method,
            amount=args.pruning_amount
        )
        
        # Make pruning permanent
        pruned_model = make_pruning_permanent(pruned_model)
        
        # Save pruned model
        pruned_model_path = os.path.join(args.output_dir, "pruned_model")
        os.makedirs(pruned_model_path, exist_ok=True)
        
        if hasattr(pruned_model, "save_pretrained"):
            pruned_model.save_pretrained(pruned_model_path)
        else:
            torch.save(pruned_model.state_dict(), os.path.join(pruned_model_path, "model.pt"))
        
        # Save processor
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(pruned_model_path)
        
        logger.info(f"Pruned model saved to {pruned_model_path}")
        
        # Benchmark pruned model
        if args.run_benchmarks:
            logger.info("Benchmarking pruned model")
            benchmark_results["pruned"] = benchmark_model(
                model=pruned_model,
                input_shape=input_shape,
                device=device,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations
            )
    
    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting model to ONNX format")
        
        # Define dynamic axes
        dynamic_axes = {
            "input": {0: "batch_size", 2: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        }
        
        # Export to ONNX
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        export_to_onnx(
            model=model,
            output_path=onnx_path,
            input_shape=input_shape,
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Model exported to ONNX: {onnx_path}")
        
        # Benchmark ONNX model
        if args.run_benchmarks:
            logger.info("Benchmarking ONNX model")
            benchmark_results["onnx"] = benchmark_onnx_model(
                onnx_path=onnx_path,
                input_shape=input_shape,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations
            )
        
        # Optimize ONNX model
        if args.optimize_onnx:
            logger.info("Optimizing ONNX model")
            
            optimized_onnx_path = os.path.join(args.output_dir, "model_optimized.onnx")
            optimize_onnx_model(
                input_path=onnx_path,
                output_path=optimized_onnx_path
            )
            
            logger.info(f"Optimized ONNX model saved to {optimized_onnx_path}")
            
            # Benchmark optimized ONNX model
            if args.run_benchmarks:
                logger.info("Benchmarking optimized ONNX model")
                benchmark_results["onnx_optimized"] = benchmark_onnx_model(
                    onnx_path=optimized_onnx_path,
                    input_shape=input_shape,
                    num_iterations=args.num_iterations,
                    warmup_iterations=args.warmup_iterations
                )
            
            # Create TensorRT engine
            if args.create_tensorrt:
                logger.info(f"Creating TensorRT engine with precision {args.tensorrt_precision}")
                
                tensorrt_path = os.path.join(args.output_dir, f"model_{args.tensorrt_precision}.engine")
                create_tensorrt_engine(
                    onnx_path=optimized_onnx_path,
                    engine_path=tensorrt_path,
                    precision=args.tensorrt_precision
                )
                
                logger.info(f"TensorRT engine saved to {tensorrt_path}")
                
                # Benchmark TensorRT engine
                if args.run_benchmarks:
                    logger.info("Benchmarking TensorRT engine")
                    benchmark_results[f"tensorrt_{args.tensorrt_precision}"] = benchmark_tensorrt_engine(
                        engine_path=tensorrt_path,
                        input_shape=input_shape,
                        num_iterations=args.num_iterations,
                        warmup_iterations=args.warmup_iterations
                    )
    
    # Save benchmark results
    if args.run_benchmarks:
        logger.info("Saving benchmark results")
        
        # Save results to JSON
        benchmark_json_path = os.path.join(args.output_dir, "benchmark_results.json")
        save_benchmark_results(
            results=benchmark_results,
            output_file=benchmark_json_path
        )
        
        # Visualize results
        benchmark_viz_path = os.path.join(args.output_dir, "benchmark_results.png")
        visualize_benchmark_results(
            results=benchmark_results,
            output_file=benchmark_viz_path
        )
        
        logger.info(f"Benchmark results saved to {benchmark_json_path}")
        logger.info(f"Benchmark visualization saved to {benchmark_viz_path}")
    
    logger.info("Optimization completed")


if __name__ == "__main__":
    main()
