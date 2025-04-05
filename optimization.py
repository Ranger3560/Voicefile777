"""
Model optimization utilities for speech recognition models.

This module contains utilities for optimizing speech recognition models,
including quantization, pruning, and TensorRT integration.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import os
import sys
import time
import numpy as np
import onnx
import json

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.whisper_model import WhisperModelWrapper, EnhancedWhisperModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to the model.
    
    Args:
        model: Model to quantize
        dtype: Quantization data type
        
    Returns:
        Quantized model
    """
    logger.info(f"Applying dynamic quantization with dtype {dtype}")
    
    # Check if model is a WhisperModelWrapper or EnhancedWhisperModel
    if isinstance(model, (WhisperModelWrapper, EnhancedWhisperModel)):
        # Get the underlying model
        if hasattr(model, "model"):
            base_model = model.model
        elif hasattr(model, "whisper"):
            base_model = model.whisper.model
        else:
            logger.warning("Could not find base model, quantizing the wrapper directly")
            base_model = model
    else:
        base_model = model
    
    # Define quantization configuration
    quantization_config = torch.quantization.default_dynamic_qconfig
    
    # Prepare model for quantization
    torch.quantization.quantize_dynamic(
        base_model,
        {nn.Linear},  # Quantize only linear layers
        dtype=dtype,
        inplace=True
    )
    
    logger.info("Dynamic quantization applied successfully")
    
    return model


def apply_static_quantization(
    model: nn.Module,
    calibration_data: torch.Tensor,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply static quantization to the model.
    
    Args:
        model: Model to quantize
        calibration_data: Data for calibration
        dtype: Quantization data type
        
    Returns:
        Quantized model
    """
    logger.info(f"Applying static quantization with dtype {dtype}")
    
    # Check if model is a WhisperModelWrapper or EnhancedWhisperModel
    if isinstance(model, (WhisperModelWrapper, EnhancedWhisperModel)):
        # Get the underlying model
        if hasattr(model, "model"):
            base_model = model.model
        elif hasattr(model, "whisper"):
            base_model = model.whisper.model
        else:
            logger.warning("Could not find base model, quantizing the wrapper directly")
            base_model = model
    else:
        base_model = model
    
    # Create a copy of the model for quantization
    quantized_model = copy.deepcopy(base_model)
    
    # Add QuantStub and DeQuantStub
    class QuantizedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant = QuantStub()
            self.model = model
            self.dequant = DeQuantStub()
        
        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x
    
    quantized_model = QuantizedModel(quantized_model)
    
    # Set model to evaluation mode
    quantized_model.eval()
    
    # Specify quantization configuration
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # Calibrate with data
    with torch.no_grad():
        for data in calibration_data:
            quantized_model(data)
    
    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    
    logger.info("Static quantization applied successfully")
    
    return quantized_model


def apply_pruning(
    model: nn.Module,
    pruning_method: str = "l1_unstructured",
    amount: float = 0.2
) -> nn.Module:
    """
    Apply pruning to the model.
    
    Args:
        model: Model to prune
        pruning_method: Pruning method (l1_unstructured, random_unstructured, etc.)
        amount: Amount of parameters to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    logger.info(f"Applying {pruning_method} pruning with amount {amount}")
    
    # Check if model is a WhisperModelWrapper or EnhancedWhisperModel
    if isinstance(model, (WhisperModelWrapper, EnhancedWhisperModel)):
        # Get the underlying model
        if hasattr(model, "model"):
            base_model = model.model
        elif hasattr(model, "whisper"):
            base_model = model.whisper.model
        else:
            logger.warning("Could not find base model, pruning the wrapper directly")
            base_model = model
    else:
        base_model = model
    
    # Apply pruning to linear layers
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            if pruning_method == "l1_unstructured":
                prune.l1_unstructured(module, name="weight", amount=amount)
            elif pruning_method == "random_unstructured":
                prune.random_unstructured(module, name="weight", amount=amount)
            else:
                logger.warning(f"Unsupported pruning method: {pruning_method}")
    
    logger.info("Pruning applied successfully")
    
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Make pruning permanent by removing pruning reparameterization.
    
    Args:
        model: Pruned model
        
    Returns:
        Model with permanent pruning
    """
    logger.info("Making pruning permanent")
    
    # Check if model is a WhisperModelWrapper or EnhancedWhisperModel
    if isinstance(model, (WhisperModelWrapper, EnhancedWhisperModel)):
        # Get the underlying model
        if hasattr(model, "model"):
            base_model = model.model
        elif hasattr(model, "whisper"):
            base_model = model.whisper.model
        else:
            logger.warning("Could not find base model, applying to the wrapper directly")
            base_model = model
    else:
        base_model = model
    
    # Make pruning permanent for all pruned layers
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass  # Not pruned
    
    logger.info("Pruning made permanent")
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Output file path
        input_shape: Input shape for tracing
        dynamic_axes: Dynamic axes configuration
        
    Returns:
        Output file path
    """
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"Model exported to ONNX successfully: {output_path}")
    
    return output_path


def optimize_onnx_model(
    input_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Optimize ONNX model.
    
    Args:
        input_path: Input ONNX model path
        output_path: Output ONNX model path
        
    Returns:
        Output file path
    """
    logger.info(f"Optimizing ONNX model: {input_path}")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = input_path.replace(".onnx", "_optimized.onnx")
    
    try:
        # Import ONNX optimizer
        from onnxruntime.transformers import optimizer
        
        # Optimize model
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type="whisper",
            num_heads=12,
            hidden_size=768
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        logger.info(f"ONNX model optimized successfully: {output_path}")
    except ImportError:
        logger.warning("onnxruntime.transformers not available, skipping optimization")
        # Copy input to output
        import shutil
        shutil.copy(input_path, output_path)
    
    return output_path


def create_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace_size: int = 1 << 30  # 1 GB
) -> str:
    """
    Create TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Input ONNX model path
        engine_path: Output TensorRT engine path
        precision: Precision mode (fp32, fp16, int8)
        workspace_size: Workspace size in bytes
        
    Returns:
        Output file path
    """
    logger.info(f"Creating TensorRT engine: {engine_path}")
    
    try:
        # Import TensorRT
        import tensorrt as trt
        
        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Create builder
        builder = trt.Builder(trt_logger)
        
        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create parser
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT ONNX parser error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # TODO: Add calibration for INT8
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine created successfully: {engine_path}")
    except ImportError:
        logger.warning("TensorRT not available, skipping engine creation")
        # Create empty file
        with open(engine_path, "wb") as f:
            f.write(b"TensorRT not available")
    
    return engine_path


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape for benchmarking
        device: Device to run benchmark on
        num_iterations: Number of iterations for benchmarking
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking model on {device} with input shape {input_shape}")
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    latency_ms = (total_time / num_iterations) * 1000
    throughput = num_iterations / total_time
    
    # Log results
    logger.info(f"Benchmark results:")
    logger.info(f"  Total time: {total_time:.4f} seconds")
    logger.info(f"  Latency: {latency_ms:.4f} ms")
    logger.info(f"  Throughput: {throughput:.4f} inferences/second")
    
    return {
        "total_time": total_time,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "device": str(device),
        "input_shape": input_shape,
        "num_iterations": num_iterations
    }


def benchmark_onnx_model(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape for benchmarking
        num_iterations: Number of iterations for benchmarking
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking ONNX model: {onnx_path}")
    
    try:
        # Import ONNX Runtime
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency_ms = (total_time / num_iterations) * 1000
        throughput = num_iterations / total_time
        
        # Log results
        logger.info(f"ONNX benchmark results:")
        logger.info(f"  Total time: {total_time:.4f} seconds")
        logger.info(f"  Latency: {latency_ms:.4f} ms")
        logger.info(f"  Throughput: {throughput:.4f} inferences/second")
        
        return {
            "total_time": total_time,
            "latency_ms": latency_ms,
            "throughput": throughput,
            "provider": session.get_providers()[0],
            "input_shape": input_shape,
            "num_iterations": num_iterations
        }
    except ImportError:
        logger.warning("ONNX Runtime not available, skipping benchmark")
        return {
            "total_time": 0.0,
            "latency_ms": 0.0,
            "throughput": 0.0,
            "provider": "none",
            "input_shape": input_shape,
            "num_iterations": 0
        }


def benchm
(Content truncated due to size limit. Use line ranges to read in chunks)