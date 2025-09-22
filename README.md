# Whisper TFLite Converter

A tool to convert Whisper models from Hugging Face Transformers to TensorFlow Lite format for efficient inference on mobile and edge devices.

## Features

- Convert any Whisper model to TFLite format
- Support for all Whisper model variants and compatible community models
- Dynamic input shape handling for different model architectures
- Built-in model validation with audio testing
- Automatic cleanup of temporary files
- Command-line interface with flexible options

## Requirements

- Linux
- Python 3.10+
- FFmpeg
- uv (for dependency management)

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# RHEL/CentOS/Fedora
sudo dnf install ffmpeg
# or
sudo yum install ffmpeg
```

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd tflite-converter

# Install dependencies with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd tflite-converter

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Convert default model (whisper-tiny)
uv run gen_model.py

# Convert specific model
uv run gen_model.py --model openai/whisper-base

# Convert with custom output path
uv run gen_model.py --model openai/whisper-large-v3 --output my-model.tflite

# Skip model testing (faster)
uv run gen_model.py --model openai/whisper-tiny --skip-test

# Test with custom audio file
uv run gen_model.py --audio path/to/audio.wav
```

### Command Line Options

- `--model`: Whisper model name from Hugging Face (default: `openai/whisper-tiny`)
- `--output`: Output TFLite model path (default: `models/{model_name}.tflite`)
- `--audio`: Audio file path for validation (default: use sample audio)
- `--skip-test`: Skip model testing after conversion

### Supported Models

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large`
- `openai/whisper-large-v2`
- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`

## Output

- TFLite models are saved to the `models/` directory
- Temporary SavedModel files are automatically cleaned up
- Model testing provides transcription validation

## Architecture

The converter:

1. Loads the Whisper model from Hugging Face
2. Creates a generation-enabled wrapper with proper input signatures
3. Exports to SavedModel format
4. Converts SavedModel to TFLite with optimizations
5. Validates the converted model with audio input
6. Cleans up temporary files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
