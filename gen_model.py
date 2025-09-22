import os
import shutil
import argparse
import tensorflow as tf
import soundfile as sf
from transformers import (
    WhisperProcessor,
    TFWhisperForConditionalGeneration,
    WhisperForConditionalGeneration
)
from datasets import load_dataset


class GenerateModel(tf.Module):
    """Generation-enabled Whisper model for TFLite conversion."""

    def __init__(self, model):
        super(GenerateModel, self).__init__()
        self.model = model
        # Get n_mels from model config
        self.n_mels = model.config.num_mel_bins

        # Create input signature based on model configuration
        self.input_signature = [
            tf.TensorSpec((1, self.n_mels, 3000), tf.float32, name="input_features"),
        ]

    @tf.function
    def serving(self, input_features):
        """Serving function for text generation from audio features."""
        outputs = self.model.generate(
            input_features,
            max_new_tokens=223,
            return_dict_in_generate=True,
        )
        return {"sequences": outputs["sequences"]}


def load_sample_audio():
    """Load sample audio data for testing."""
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return ds[0]["audio"]["array"]


def load_audio_file(audio_path):
    """Load audio from file."""
    audio_data, _ = sf.read(audio_path)
    return audio_data


def convert_to_tflite(model_name="openai/whisper-tiny", tflite_path=None):
    """Convert Whisper model to TFLite format."""

    try:
        # Set default output path to models/ directory
        if tflite_path is None:
            model_filename = model_name.split("/")[-1] + ".tflite"
            tflite_path = f"models/{model_filename}"

        # Create directories
        os.makedirs("temp", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        saved_model_dir = "./temp/tf_whisper_saved_generate"

        # Load processor and model
        print(f"Loading model: {model_name}")
        processor = WhisperProcessor.from_pretrained(model_name)

        # Try to load TensorFlow model from PyTorch
        try:
            model = TFWhisperForConditionalGeneration.from_pretrained(model_name, from_pt=True)
        except OSError:
            # PyTorch model not found, load via PyTorch and convert
            print("PyTorch model not found, loading via PyTorch and converting...")

            # Load PyTorch model
            pt_model = WhisperForConditionalGeneration.from_pretrained(model_name)

            # Save PyTorch model temporarily
            temp_pt_dir = "./temp/pytorch_model"
            os.makedirs(temp_pt_dir, exist_ok=True)
            pt_model.save_pretrained(temp_pt_dir)

            # Load TensorFlow model from saved PyTorch model
            model = TFWhisperForConditionalGeneration.from_pretrained(temp_pt_dir, from_pt=True)

            # Clean up temporary PyTorch model
            shutil.rmtree(temp_pt_dir)

        # Create generation-enabled model
        print("Creating generation-enabled model...")
        generate_model = GenerateModel(model=model)

        # Save as SavedModel
        print(f"Saving SavedModel to {saved_model_dir}")
        # Apply input signature to serving function
        concrete_function = generate_model.serving.get_concrete_function(*generate_model.input_signature)
        tf.saved_model.save(
            generate_model,
            saved_model_dir,
            signatures={"serving_default": concrete_function}
        )

        # Convert to TFLite
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save TFLite model
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"Generation-enabled TFLite model saved at {tflite_path}")

        # Clean up temp directory
        print("Cleaning up temporary files...")
        if os.path.exists("temp"):
            shutil.rmtree("temp")

        return tflite_path, processor

    except Exception as e:
        print(f"Error during TFLite model conversion: {e}")
        # Clean up temp directory on error
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        raise


def test_model(tflite_path, processor, audio_path=None):
    """Test the TFLite model with audio input."""

    try:
        # Load audio data
        if audio_path:
            print(f"Loading audio from: {audio_path}")
            audio_data = load_audio_file(audio_path)
        else:
            print("Loading sample audio data...")
            audio_data = load_sample_audio()

        # Process audio to get input features
        inputs = processor(audio_data, return_tensors="tf")
        input_features = inputs.input_features
        print(f"Input features shape: {input_features.shape}")

        # Test TFLite model
        print("Testing TFLite model...")
        interpreter = tf.lite.Interpreter(tflite_path)
        interpreter.allocate_tensors()

        tflite_generate = interpreter.get_signature_runner()
        generated_ids = tflite_generate(input_features=input_features)["sequences"]
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("TFLite model transcription:", transcription)

        return transcription

    except Exception as e:
        print(f"Error during model testing: {e}")
        raise


def create_tflite_model(model_name="openai/whisper-tiny", tflite_path=None, audio_path=None, skip_test=False):
    """Create and optionally test a TFLite model for Whisper speech recognition."""

    # Convert model to TFLite
    tflite_path, processor = convert_to_tflite(model_name, tflite_path)

    # Test model if not skipped
    if not skip_test:
        test_model(tflite_path, processor, audio_path)

    return tflite_path


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description="Convert Whisper model to TFLite format")
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="Whisper model name (default: openai/whisper-tiny)"
    )
    parser.add_argument(
        "--output",
        help="Output TFLite model path (default: models/{model_name}.tflite)"
    )
    parser.add_argument(
        "--audio",
        help="Audio file path for validation (default: use sample audio)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip model testing after conversion"
    )

    args = parser.parse_args()

    create_tflite_model(
        model_name=args.model,
        tflite_path=args.output,
        audio_path=args.audio,
        skip_test=args.skip_test
    )


if __name__ == "__main__":
    main()
