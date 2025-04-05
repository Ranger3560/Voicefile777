# Environment Requirements

This document outlines the environment requirements for running the speech recognition model project. These requirements apply to both local environments and cloud environments like Google Colab.

## Table of Contents

1. [Python Requirements](#python-requirements)
2. [Hardware Requirements](#hardware-requirements)
3. [Cloud Environment Setup](#cloud-environment-setup)
4. [Local Environment Setup](#local-environment-setup)
5. [Docker Environment](#docker-environment)

## Python Requirements

### Core Dependencies

The following core dependencies are required to run the speech recognition model:

```
torch>=1.12.0
torchaudio>=0.12.0
transformers>=4.25.0
datasets>=2.8.0
librosa>=0.9.2
soundfile>=0.11.0
evaluate>=0.3.0
jiwer>=2.5.1
```

### Optional Dependencies

The following optional dependencies are recommended for additional functionality:

```
tensorboard>=2.11.0
matplotlib>=3.6.0
numpy>=1.23.0
pandas>=1.5.0
seaborn>=0.12.0
wandb>=0.13.0
onnx>=1.13.0
onnxruntime>=1.13.0
```

### Optimization Dependencies

The following dependencies are required for model optimization:

```
onnx>=1.13.0
onnxruntime>=1.13.0
onnxruntime-gpu>=1.13.0  # For GPU acceleration
```

For TensorRT integration (optional):
```
tensorrt>=8.4.0
pycuda>=2022.1
```

### Full Requirements File

A complete `requirements.txt` file is provided below:

```
torch>=1.12.0
torchaudio>=0.12.0
transformers>=4.25.0
datasets>=2.8.0
librosa>=0.9.2
soundfile>=0.11.0
evaluate>=0.3.0
jiwer>=2.5.1
tensorboard>=2.11.0
matplotlib>=3.6.0
numpy>=1.23.0
pandas>=1.5.0
seaborn>=0.12.0
wandb>=0.13.0
onnx>=1.13.0
onnxruntime>=1.13.0
```

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk Space**: 10 GB
- **GPU**: None (CPU-only inference)

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Disk Space**: 50+ GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (e.g., NVIDIA RTX 3070 or better)
- **CUDA**: CUDA 11.6+ and cuDNN 8.4+

### Cloud GPU Requirements

For training on cloud platforms like Google Colab:

- **Google Colab**: T4 or P100 GPU (available in free tier)
- **Google Colab Pro**: V100 or A100 GPU

## Cloud Environment Setup

### Google Colab

Google Colab provides a free cloud environment with GPU support. To set up Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Set up GPU acceleration:
   - Click on "Runtime" in the menu
   - Select "Change runtime type"
   - Set "Hardware accelerator" to "GPU"
   - Click "Save"
4. Install dependencies:
   ```python
   !pip install torch torchaudio transformers datasets librosa soundfile evaluate jiwer tensorboard matplotlib numpy pandas
   ```
5. Clone the repository:
   ```python
   !git clone https://github.com/your-username/speech_model_project.git
   %cd speech_model_project
   ```

For detailed instructions, see [Google Colab Setup Instructions](colab_setup.md).

### AWS SageMaker

To set up AWS SageMaker:

1. Create a SageMaker notebook instance with GPU support
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech_model_project.git
   cd speech_model_project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Azure Machine Learning

To set up Azure Machine Learning:

1. Create an Azure ML compute instance with GPU support
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech_model_project.git
   cd speech_model_project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Local Environment Setup

### Using pip

To set up a local environment using pip:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech_model_project.git
   cd speech_model_project
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using conda

To set up a local environment using conda:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech_model_project.git
   cd speech_model_project
   ```
2. Create a conda environment:
   ```bash
   conda create -n speech_model python=3.10
   conda activate speech_model
   ```
3. Install PyTorch with GPU support:
   ```bash
   conda install pytorch torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Docker Environment

A Dockerfile is provided to create a containerized environment:

```dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Command to run when container starts
CMD ["bash"]
```

To build and run the Docker container:

1. Build the Docker image:
   ```bash
   docker build -t speech_model .
   ```
2. Run the Docker container:
   ```bash
   docker run --gpus all -it speech_model
   ```

For training:
```bash
docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output speech_model python scripts/train.py --output_dir=/app/output
```

For evaluation:
```bash
docker run --gpus all -it -v $(pwd)/output:/app/output speech_model python scripts/evaluate.py --model_path=/app/output/final_model --output_dir=/app/evaluation_results
```
