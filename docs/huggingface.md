# Hugging Face Models in Heihachi

This guide provides instructions on using specialized AI models from Hugging Face in the Heihachi audio analysis framework.

## Installation

To use the HuggingFace integration, you need to install the required dependencies:

```bash
pip install -r requirements-huggingface.txt
```

## Available Models

Heihachi integrates the following specialized models:

### Feature Extraction

- **BEATs Models** (microsoft/BEATs): Bidirectional audio encoders that provide rich temporal and spectral embeddings
- **Whisper Encoder** (openai/whisper): Robust speech and non-speech feature extraction

### Audio Source Separation

- **Demucs** (facebook/demucs): Separates audio into stems (drums, bass, vocals, other)

### Beat Detection

- **Beat-Transformer** (nicolaus625/cmi): Beat and downbeat detection with high accuracy

## Command-Line Usage

### Feature Extraction

Extract features from an audio file using the BEATs model:

```bash
python -m src.main hf extract path/to/audio.mp3 --output features.json
```

Options:
- `--model`: Specify model name (default: "microsoft/BEATs-base")
- `--output`: Path to save extracted features (JSON format)
- `--max-length`: Maximum audio length to process in seconds (default: 30)
- `--cpu`: Force CPU inference even if GPU is available

### Stem Separation

Separate an audio file into stems:

```bash
python -m src.main hf separate path/to/audio.mp3 --output-dir ./stems --save-stems
```

Options:
- `--model`: Specify model name (default: "facebook/demucs")
- `--output-dir`: Directory to save separated stems
- `--num-stems`: Number of stems to separate (4 or 6, default: 4)
- `--save-stems`: Enable to save stems to disk

### Beat Detection

Detect beats and downbeats in an audio file:

```bash
python -m src.main hf beats path/to/audio.mp3 --output beats.json
```

Options:
- `--model`: Specify model name (default: "nicolaus625/cmi")
- `--output`: Path to save beat data or visualization
- `--visualize`: Generate visualization of detected beats
- `--no-downbeats`: Skip downbeat detection

## Python API Usage

You can also use these models directly in your Python code:

```python
from heihachi.huggingface import FeatureExtractor, StemSeparator, BeatDetector

# Extract features
extractor = FeatureExtractor(model="microsoft/BEATs-base")
features = extractor.extract(audio_path="track.mp3")

# Separate stems
separator = StemSeparator()
stems = separator.separate(audio_path="track.mp3")
drums = stems["drums"]
bass = stems["bass"]

# Detect beats
detector = BeatDetector()
beats = detector.detect(audio_path="track.mp3", visualize=True, output_path="beats.png")
print(f"Tempo: {beats['tempo']} BPM")
```

## Performance Considerations

- GPU acceleration is enabled by default when available
- For large audio files, batched processing is automatically used
- Memory usage can be high for some models - consider using a machine with at least 8GB RAM

## HuggingFace API Keys

For models that require authentication:

```bash
python -m src.main hf extract path/to/audio.mp3 --api-key YOUR_HF_API_KEY
```

You can also set the environment variable `HF_API_TOKEN` to avoid passing it on the command line.

## Extending with New Models

The framework is designed to be easily extended with other HuggingFace models. See the developer documentation for more information on adding new model types. 