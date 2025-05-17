<h1 align="center">Heihachi</h1>
<p align="center"><em> what makes a tiger so strong is that it lacks humanity</em></p>

<p align="center">
  <img src="./heihachi.png" alt="Heihachi Logo" width="300"/>
</p>

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Heihachi Audio Analysis Framework

Advanced audio analysis framework for processing, analyzing, and visualizing audio files with optimized performance.

## Features

- High-performance audio file processing
- Batch processing for handling multiple files
- Memory optimization for large audio files
- Parallel processing capabilities
- Visualization tools for spectrograms and waveforms
- Interactive results exploration with command-line and web interfaces
- Progress tracking for long-running operations
- Export options in multiple formats (JSON, CSV, YAML, etc.)
- Comprehensive CLI with shell completion
- HuggingFace integration for advanced audio analysis and neural processing

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/heihachi.git
cd heihachi

# Run the setup script
python scripts/setup.py
```

### Options

The setup script supports several options:

```
--install-dir DIR     Installation directory
--dev                 Install development dependencies
--no-gpu              Skip GPU acceleration dependencies
--no-interactive      Skip interactive mode dependencies
--shell-completion    Install shell completion scripts
--no-confirm          Skip confirmation prompts
--venv                Create and use a virtual environment
--venv-dir DIR        Virtual environment directory (default: .venv)
```

### Manual Installation

If you prefer to install manually:

```bash
# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```bash
# Process a single audio file
heihachi process audio.wav --output results/

# Process a directory of audio files
heihachi process audio_dir/ --output results/

# Batch processing with different configurations
heihachi batch audio_dir/ --config configs/performance.yaml
```

### Interactive Mode

```bash
# Start interactive command-line explorer with processed results
heihachi interactive --results-dir results/

# Start web-based interactive explorer
heihachi interactive --web --results-dir results/

# Compare multiple results with interactive explorer
heihachi compare results1/ results2/

# Show only progress demo
heihachi demo --progress-demo
```

### Export Options

```bash
# Export results to different formats
heihachi export results/ --format json
heihachi export results/ --format csv
heihachi export results/ --format markdown
```

## HuggingFace Integration

Heihachi integrates specialized AI models from Hugging Face, enabling advanced neural processing of audio using state-of-the-art models. This integration follows a structured implementation approach with models carefully selected for electronic music analysis tasks.

### Available Models

The following specialized audio analysis models are available:

| Category | Model Type | Default Model | Description | Priority |
|----------|------------|---------------|-------------|----------|
| **Core Feature Extraction** | Generic spectral + temporal embeddings | [microsoft/BEATs](https://huggingface.co/microsoft/BEATs) | Bidirectional ViT-style encoder trained with acoustic tokenisers; provides 768-d latent at ~20 ms hop | High |
| | Robust speech & non-speech features | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Trained on >5M hours; encoder provides 1280-d features tracking energy, voicing & language | High |
| **Audio Source Separation** | Stem isolation | [Demucs v4](https://huggingface.co/spaces/abidlabs/music-separation) | Returns 4-stem or 6-stem tensors for component-level analysis | High |
| **Rhythm Analysis** | Beat / down-beat tracking | [Beat-Transformer](https://huggingface.co/nicolaus625/cmi) | Dilated self-attention encoder with F-measure ~0.86 | High |
| | Low-latency beat-tracking | [BEAST](https://github.com/beats-team/beast) | 50 ms latency, causal attention; ideal for real-time DJ analysis | Medium |
| | Drum-onset / kit piece ID | [DunnBC22/wav2vec2-base-Drum_Kit_Sounds](https://huggingface.co/DunnBC22/wav2vec2-base-Drum_Kit_Sounds) | Fine-tuned on kick/snare/tom/overhead labels | Medium |
| **Multimodal & Similarity** | Multimodal similarity / tagging | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | Query with free-text and compute cosine similarity on 512-d embeddings | Medium |
| | Zero-shot tag & prompt embedding | [UniMus/OpenJMLA](https://huggingface.co/UniMus/OpenJMLA) | Score arbitrary tag strings for effect-chain heuristics | Medium |
| **Future Extensions** | Audio captioning | [slseanwu/beats-conformer-bart-audio-captioner](https://huggingface.co/slseanwu/beats-conformer-bart-audio-captioner) | Produces textual descriptions per segment | Low |
| | Similarity retrieval UI | CLAP embeddings + FAISS | Index embeddings and expose nearest-neighbor search | Low |

### Configuration

Configure HuggingFace models in `configs/huggingface.yaml`:

```yaml
# Enable/disable HuggingFace integration
enabled: true

# API key for accessing HuggingFace models (leave empty to use public models only)
api_key: ""

# Specialized model settings
feature_extraction:
  enabled: true
  model: "microsoft/BEATs-base"

beat_detection:
  enabled: true
  model: "nicolaus625/cmi"

# Additional models (disabled by default to save resources)
drum_sound_analysis:
  enabled: false
  model: "DunnBC22/wav2vec2-base-Drum_Kit_Sounds"

similarity:
  enabled: false
  model: "laion/clap-htsat-fused"

# See configs/huggingface.yaml for all available options
```

### Installation

To use the HuggingFace integration, you need to install the required dependencies:

```bash
pip install -r requirements-huggingface.txt
```

### Command-Line Usage

#### Feature Extraction

Extract features from an audio file using the BEATs model:

```bash
python -m src.main hf extract path/to/audio.mp3 --output features.json
```

Options:
- `--model`: Specify model name (default: "microsoft/BEATs-base")
- `--output`: Path to save extracted features (JSON format)
- `--max-length`: Maximum audio length to process in seconds (default: 30)
- `--cpu`: Force CPU inference even if GPU is available

#### Stem Separation

Separate an audio file into stems:

```bash
python -m src.main hf separate path/to/audio.mp3 --output-dir ./stems --save-stems
```

Options:
- `--model`: Specify model name (default: "facebook/demucs")
- `--output-dir`: Directory to save separated stems
- `--num-stems`: Number of stems to separate (4 or 6, default: 4)
- `--save-stems`: Enable to save stems to disk

#### Beat Detection

Detect beats and downbeats in an audio file:

```bash
python -m src.main hf beats path/to/audio.mp3 --output beats.json
```

Options:
- `--model`: Specify model name (default: "nicolaus625/cmi")
- `--output`: Path to save beat data or visualization
- `--visualize`: Generate visualization of detected beats
- `--no-downbeats`: Skip downbeat detection

#### Other Commands

| Command | Description | Example |
|---------|-------------|---------|
| `extract` | Extract features using neural models | `python -m src.main hf extract audio.wav` |
| `analyze-drums` | Analyze drum sounds | `python -m src.main hf analyze-drums audio.wav --visualize` |
| `drum-patterns` | Detect and analyze drum patterns | `python -m src.main hf drum-patterns audio.wav --mode pattern` |
| `tag` | Perform zero-shot audio tagging | `python -m src.main hf tag audio.wav --categories "genre:techno,house,ambient"` |
| `caption` | Generate audio descriptions | `python -m src.main hf caption audio.wav --mix-notes` |
| `similarity` | Audio-text similarity analysis | `python -m src.main hf similarity audio.wav --mode timestamps --query "bass drop"` |
| `realtime-beats` | Real-time beat tracking | `python -m src.main hf realtime-beats --file --input audio.wav` |

### Python API Usage

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

### Implementation Plan

The integration of specialized models follows a phased approach:

1. **Phase 1: Core Models Integration**
   - Integrate BEATs models for feature extraction
   - Add Whisper encoder for robust feature analysis
   - Implement Demucs for stem separation
   - Add Beat-Transformer for rhythm analysis

2. **Phase 2: Specialized Analysis**
   - Implement CLAP for multimodal similarity
   - Integrate wav2vec2 for drum sound analysis
   - Add BEAST for real-time beat tracking

3. **Phase 3: Advanced Features**
   - Implement OpenJMLA for zero-shot tagging
   - Add audio captioning capabilities
   - Build similarity retrieval interface

### Performance Considerations

- GPU acceleration is enabled by default when available
- For large audio files, batched processing is automatically used
- Memory usage can be high for some models - consider using a machine with at least 8GB RAM
- HuggingFace API keys can be provided for models requiring authentication

## Interactive Explorer and Progress Utilities

### Interactive Explorer

Heihachi provides two interactive interfaces for exploring analysis results:

#### Command-Line Explorer

The command-line explorer allows you to interactively explore audio analysis results:

```bash
python -m src.cli.interactive_cli --results-dir /path/to/results
```

Available commands:
- `list [pattern]`: List available analysis result files
- `open INDEX/FILENAME`: Open an analysis result file
- `summary`: Show summary of the current analysis result
- `info FEATURE_NAME`: Show detailed information about a specific feature
- `plot TYPE [FEATURE_NAME] [--save FILENAME]`: Plot a feature or visualization
- `compare FEATURE_NAME [file1 file2 ...]`: Compare a feature across multiple results
- `distribution FEATURE_NAME [file1 file2 ...]`: Plot distribution of a feature
- `export FORMAT [FILENAME]`: Export the current result to another format
- `refresh`: Refresh the list of available result files
- `help`: Show available commands
- `exit`, `quit`: Exit the explorer

Plot types include `waveform`, `spectrogram`, and `feature FEATURE_NAME`.

Export formats include `csv`, `json`, and `markdown`.

#### Web UI Explorer

A browser-based interface for visualizing and exploring analysis results:

```bash
python -m src.cli.interactive_cli --web --result-file /path/to/result.json
```

Options:
- `--web`: Start web UI instead of CLI explorer
- `--host`: Host to bind the web server to (default: `localhost`)
- `--port`: Port to bind the web server to (default: `5000`)
- `--result-file`: Specific result file to load in web UI
- `--no-browser`: Don't automatically open browser

### Progress Utilities

The framework includes several utilities for tracking progress in long-running operations:

#### Progress Context

```python
from src.utils.progress import progress_context

with progress_context("Processing audio", total=100) as progress:
    for i in range(100):
        # Do some work
        progress.update(1)
```

#### Progress Manager

For tracking multiple operations simultaneously:

```python
from src.utils.progress import ProgressManager

manager = ProgressManager()
with manager.task("Processing audio", total=100) as task1:
    with manager.task("Extracting features", total=50) as task2:
        # Operations with nested progress tracking
```

#### CLI Progress Bar

For simple command-line progress indication:

```python
from src.utils.progress import cli_progress_bar

for file in cli_progress_bar(files, desc="Processing files"):
    # Process each file
```

### Batch Processing

Heihachi includes a batch processing system for analyzing multiple audio files with different configurations.

#### Batch Processing CLI

```bash
python -m src.cli.batch_cli --input-dir /path/to/audio --config /path/to/config.json
```

Options:
- `--input-dir`: Directory containing audio files to process
- `--batch-file`: JSON file containing batch processing specification
- `--pattern`: File pattern to match (default: `*.wav`)
- `--max-files`: Maximum number of files to process
- `--config`: Configuration file(s) to use
- `--output-dir`: Directory to save results (default: `results`)
- `--export`: Export formats (default: `json`)
- `--parallel`: Process files in parallel (default: `True`)
- `--no-parallel`: Disable parallel processing
- `--workers`: Number of worker processes
- `--progress`: Progress display type (`bar`, `rich`, `simple`, `none`)
- `--log-level`: Set logging level
- `--quiet`: Minimize output (overrides `--progress`)

#### Batch Configuration

Batch processing can be configured through a JSON file:

```json
{
  "output_dir": "results",
  "configs": [
    {
      "name": "config1",
      "path": "configs/config1.json",
      "file_patterns": ["*.wav", "*.mp3"]
    },
    {
      "name": "config2",
      "path": "configs/config2.json",
      "files": ["file1.wav", "file2.wav"]
    }
  ],
  "input_dirs": ["audio1", "audio2"],
  "parallel": true,
  "max_workers": 4
}
```

### Demonstration

A demonstration script is available to showcase the interactive explorer and progress utilities:

```bash
python scripts/interactive_demo.py
```

This script demonstrates:
1. Progress tracking during sample audio processing
2. Interactive exploration of analysis results
3. Various visualization capabilities

Options:
- `--skip-processing`: Skip processing and use existing results
- `--progress-demo`: Show only progress indicators demonstration

## Advanced Usage

### Custom Pipelines

```yaml
# Example pipeline configuration (pipeline.yaml)
stages:
  - name: load
    processor: AudioLoader
    params:
      sample_rate: 44100
  
  - name: analyze
    processor: SpectrumAnalyzer
    params:
      n_fft: 2048
      hop_length: 512
  
  - name: extract
    processor: FeatureExtractor
    params:
      features: [mfcc, chroma, tonnetz]
```

Run with custom pipeline:
```bash
heihachi process audio.wav --pipeline pipeline.yaml
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_audio_processing.py
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Check typing
mypy src/
```

## License

MIT

# Heihachi: Neural Processing of Electronic Music

## Overview

Heihachi is a high-performance audio analysis framework designed for electronic music, with a particular focus on neurofunk and drum & bass genres. The system implements novel approaches to audio analysis by combining neurological models of rhythm processing with advanced signal processing techniques.

## Theoretical Foundation

### Neural Basis of Rhythm Processing

The framework is built upon established neuroscientific research demonstrating that humans possess an inherent ability to synchronize motor responses with external rhythmic stimuli. This phenomenon, known as beat-based timing, involves complex interactions between auditory and motor systems in the brain.

Key neural mechanisms include:

1. **Beat-based Timing Networks**
   - Basal ganglia-thalamocortical circuits
   - Supplementary motor area (SMA)
   - Premotor cortex (PMC)

2. **Temporal Processing Systems**
   - Duration-based timing mechanisms
   - Beat-based timing mechanisms
   - Motor-auditory feedback loops

### Motor-Auditory Coupling

Research has shown that low-frequency neural oscillations from motor planning areas guide auditory sampling, expressed through coherence measures:

$$
C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f)S_{yy}(f)}
$$

Where:
- $C_{xy}(f)$ represents coherence at frequency $f$
- $S_{xy}(f)$ is the cross-spectral density
- $S_{xx}(f)$ and $S_{yy}(f)$ are auto-spectral densities

### Extended Mathematical Framework

In addition to the coherence measures, we utilize several key mathematical formulas:

1. **Spectral Decomposition**:
For analyzing sub-bass and Reese bass components:

$$
X(k) = \sum_{n=0}^{N-1} x(n)e^{-j2\pi kn/N}
$$

2. **Groove Pattern Analysis**:
For microtiming deviations:

$$
MT(n) = \frac{1}{K}\sum_{k=1}^{K} |t_k(n) - t_{ref}(n)|
$$

3. **Amen Break Detection**:
Pattern matching score:

$$
S_{amen}(t) = \sum_{f} w(f)|X(f,t) - A(f)|^2
$$

4. **Reese Bass Analysis**:
For analyzing modulation and phase relationships:

$$
R(t,f) = \left|\sum_{k=1}^{K} A_k(t)e^{j\phi_k(t)}\right|^2
$$

Where:
- $A_k(t)$ is the amplitude of the k-th component
- $\phi_k(t)$ is the phase of the k-th component

5. **Transition Detection**:
For identifying mix points and transitions:

$$
T(t) = \alpha\cdot E(t) + \beta\cdot S(t) + \gamma\cdot H(t)
$$

Where:
- $E(t)$ is energy change
- $S(t)$ is spectral change
- $H(t)$ is harmonic change
- $\alpha, \beta, \gamma$ are weighting factors

6. **Similarity Computation**:
For comparing audio segments:

$$
Sim(x,y) = \frac{\sum_i w_i \cdot sim_i(x,y)}{\sum_i w_i}
$$

Where:
- $sim_i(x,y)$ is the similarity for feature i
- $w_i$ is the weight for feature i

7. **Segment Clustering**:
Using DBSCAN with adaptive distance:

$$
D(p,q) = \sqrt{\sum_{i=1}^{N} \lambda_i(f_i(p) - f_i(q))^2}
$$

Where:
- $f_i(p)$ is feature i of point p
- $\lambda_i$ is the importance weight for feature i

### Additional Mathematical Framework

8. **Bass Design Analysis**:
For analyzing Reese bass modulation depth:

$$
M(t) = \frac{max(A(t)) - min(A(t))}{max(A(t)) + min(A(t))}
$$

9. **Effect Chain Detection**:
For compression ratio estimation:

$$
CR(x) = \frac{\Delta_{in}}{\Delta_{out}} = \frac{x_{in,2} - x_{in,1}}{x_{out,2} - x_{out,1}}
$$

10. **Pattern Recognition**:
For rhythmic similarity using dynamic time warping:

$$
DTW(X,Y) = min\left(\sum_{k=1}^K w_k \cdot d(x_{i_k}, y_{j_k})\right)
$$

11. **Transition Analysis**:
For blend detection using cross-correlation:

$$
R_{xy}(\tau) = \sum_{n=-\infty}^{\infty} x(n)y(n+\tau)
$$

## Core Components

### 1. Feature Extraction Pipeline

#### Rhythmic Analysis
- Automated drum pattern recognition
- Groove quantification
- Microtiming analysis
- Syncopation detection

#### Spectral Analysis
- Multi-band decomposition
- Harmonic tracking
- Timbral feature extraction
- Sub-bass characterization

#### Component Analysis
- Sound source separation
- Transformation detection
- Energy distribution analysis
- Component relationship mapping

### 2. Alignment Modules

#### Amen Break Analysis
- Pattern matching and variation detection
- Transformation identification
- Groove characteristic extraction
- VIP/Dubplate classification
- Robust onset envelope extraction with fault tolerance
- Dynamic time warping with optimal window functions

#### Prior Subspace Analysis
- Neurofunk-specific component separation
- Bass sound design analysis
- Effect chain detection
- Temporal structure analysis

#### Composite Similarity
- Multi-band similarity computation
- Transformation-aware comparison
- Groove-based alignment
- Confidence scoring

### 3. Annotation System

#### Peak Detection
- Multi-band onset detection
- Adaptive thresholding
- Feature-based peak classification
- Confidence scoring

#### Segment Clustering
- Pattern-based segmentation
- Hierarchical clustering
- Relationship analysis
- Transition detection

#### Transition Detection
- Mix point identification
- Blend type classification
- Energy flow analysis
- Structure boundary detection

### 4. Robust Processing Framework

#### Error Handling and Validation
- Empty audio detection and graceful recovery
- Sample rate validation and default fallbacks
- Signal integrity verification
- Automatic recovery mechanisms

#### Memory Management
- Streaming processing for large files
- Resource optimization and monitoring
- Garbage collection optimization
- Chunked processing of large audio files

#### Signal Processing Enhancements
- Proper window functions to eliminate spectral leakage
- Normalized processing paths
- Adaptive parameters based on content
- Fault-tolerant alignment algorithms

### Extended Pipeline Architecture

```mermaid
graph LR
    A[Audio Stream] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Component Analysis]
    D --> E[Pattern Recognition]
    E --> F[Result Generation]
    
    subgraph "Feature Extraction"
    C1[Spectral] --> C2[Temporal]
    C2 --> C3[Rhythmic]
    end
```

## Command Line Interface (CLI)

Heihachi provides a powerful command-line interface to analyze audio files. The CLI allows you to process individual audio files or entire directories of audio files.

### Installation

First, ensure the package is installed:

```bash
# Clone the repository
git clone https://github.com/your-username/heihachi.git
cd heihachi

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

The basic command structure is:

```bash
python -m src.main [input_file] [options]
```

Where `[input_file]` can be either a single audio file or a directory containing multiple audio files.

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_file` | Path to audio file or directory (required) | - |
| `-c, --config` | Path to configuration file | ../configs/default.yaml |
| `-o, --output` | Path to output directory | ../results |
| `--cache-dir` | Path to cache directory | ../cache |
| `-v, --verbose` | Enable verbose logging | False |

### Examples

#### Process a single audio file:

```bash
python -m src.main /path/to/track.wav
```

#### Process an entire directory of audio files:

```bash
python -m src.main /path/to/audio/folder
```

#### Use a custom configuration file:

```bash
python -m src.main /path/to/track.wav -c /path/to/custom_config.yaml
```

#### Specify custom output directory:

```bash
python -m src.main /path/to/track.wav -o /path/to/custom_output
```

#### Enable verbose logging:

```bash
python -m src.main /path/to/track.wav -v
```

### Processing Results

After processing, the results are saved to the output directory (default: `../results`). For each audio file, the following is generated:

1. **Analysis data**: JSON files containing detailed analysis results
2. **Visualizations**: Graphs and plots showing various aspects of the audio analysis
3. **Summary report**: Overview of the key findings and detected patterns

### Amen Break Analysis

For specific Amen break analysis, ensure the reference Amen break sample is available in the `../public/amen_break.wav` path. The analysis will detect Amen break variations, their timing, and provide confidence scores.

Example for specifically analyzing Amen breaks:

```bash
python -m src.main /path/to/jungle_track.wav
```

The output will indicate if Amen break patterns were detected, along with timestamps and variation information.

### Neurofunk Component Analysis

```mermaid
graph TD
    A[Input Signal] --> B[Sub-bass Extraction]
    A --> C[Reese Detection]
    A --> D[Drum Pattern Analysis]
    B --> E[Bass Pattern]
    C --> E
    D --> F[Rhythm Grid]
    E --> G[Component Fusion]
    F --> G
```

### Feature Extraction Pipeline

```mermaid
graph TD
    A[Audio Input] --> B[Preprocessing]
    B --> C[Feature Extraction]
    
    subgraph "Feature Extraction"
        C1[Spectral Analysis] --> D1[Sub-bass]
        C1 --> D2[Mid-range]
        C1 --> D3[High-freq]
        
        C2[Temporal Analysis] --> E1[Envelope]
        C2 --> E2[Transients]
        
        C3[Rhythmic Analysis] --> F1[Beats]
        C3 --> F2[Patterns]
    end
    
    D1 --> G[Feature Fusion]
    D2 --> G
    D3 --> G
    E1 --> G
    E2 --> G
    F1 --> G
    F2 --> G
```

### Annotation System Flow

```mermaid
graph LR
    A[Audio Stream] --> B[Peak Detection]
    B --> C[Segment Creation]
    C --> D[Pattern Analysis]
    D --> E[Clustering]
    
    subgraph "Pattern Analysis"
        D1[Drum Patterns]
        D2[Bass Patterns]
        D3[Effect Patterns]
    end
```

### Audio Scene Analysis

```mermaid
graph TD
    A[Input Signal] --> B[Background Separation]
    A --> C[Foreground Analysis]
    
    subgraph "Background"
        B1[Ambient Detection]
        B2[Noise Floor]
        B3[Reverb Tail]
    end
    
    subgraph "Foreground"
        C1[Transient Detection]
        C2[Note Events]
        C3[Effect Events]
    end
```

### Result Fusion Process

```mermaid
graph LR
    A[Component Results] --> B[Confidence Scoring]
    B --> C[Weight Assignment]
    C --> D[Fusion]
    D --> E[Final Results]
    
    subgraph "Confidence Scoring"
        B1[Pattern Confidence]
        B2[Feature Confidence]
        B3[Temporal Confidence]
    end
```

### Neurofunk-Specific Analysis

#### Bass Sound Design Analysis
```python
1. Reese Bass Components:
   - Fundamental frequency tracking
   - Phase relationship analysis
   - Modulation pattern detection
   - Harmonic content analysis

2. Sub Bass Characteristics:
   - Frequency range: 20-60 Hz
   - Envelope characteristics
   - Distortion analysis
   - Phase alignment
```

#### Effect Chain Detection
```python
1. Signal Chain Analysis:
   - Compression detection
   - Distortion identification
   - Filter resonance analysis
   - Modulation effects

2. Processing Order:
   - Pre/post processing detection
   - Parallel processing identification
   - Send/return effect analysis
```

#### Pattern Transformation Analysis
```python
1. Rhythmic Transformations:
   - Time-stretching detection
   - Beat shuffling analysis
   - Groove template matching
   - Syncopation patterns

2. Spectral Transformations:
   - Frequency shifting
   - Harmonic manipulation
   - Formant preservation
   - Resynthesis detection
```

## Implementation Details

### Audio Processing Pipeline

1. **Preprocessing**
   ```python
   - Sample rate normalization (44.1 kHz)
   - Stereo to mono conversion when needed
   - Segment-wise processing for large files
   - Empty audio detection and validation
   - Signal integrity verification
   ```

2. **Feature Extraction**
   ```python
   - Multi-threaded processing
   - GPU acceleration where available
   - Efficient memory management
   - Caching system for intermediate results
   - Hann window application to prevent spectral leakage
   ```

3. **Analysis Flow**
   ```python
   - Cascading analysis system
   - Component-wise processing
   - Result fusion and validation
   - Confidence scoring
   - Graceful error handling and recovery
   ```

### Performance Optimizations

1. **Memory Management**
   - Streaming processing for large files
   - Efficient cache utilization
   - GPU memory optimization
   - Automatic garbage collection optimization
   - Chunked loading for very large files
   - Audio validation at each processing stage

2. **Parallel Processing**
   - Multi-threaded feature extraction
   - Batch processing capabilities
   - Distributed analysis support
   - Adaptive resource allocation
   - Scalable parallel execution

3. **Storage Efficiency**
   - Compressed result storage
   - Metadata indexing
   - Version control for analysis results
   - Simple, consistent path handling

### Error Handling and Recovery

1. **Robust Processing**
   ```python
   - Validation of audio signals before processing
   - Fallback mechanisms for empty or corrupted audio
   - Graceful recovery from processing errors
   - Comprehensive logging of failure points
   ```

2. **Data Integrity**
   ```python
   - Sample rate validation and automatic correction
   - Empty array detection and prevention
   - Default parameter settings when configuration is missing
   - Prevention of null pointer exceptions in signal processing chain
   ```

3. **Processing Resilience**
   ```python
   - Windowing functions to eliminate warnings and improve spectral quality
   - Automatic memory management for very large files
   - Adaptive parameter selection based on file size
   - Simplified path handling with hardcoded relative paths
   ```

### Performance Metrics

For evaluating analysis accuracy:

$$
Accuracy_{component} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:
- TP: True Positives (correctly identified patterns)
- TN: True Negatives (correctly rejected non-patterns)
- FP: False Positives (incorrectly identified patterns)
- FN: False Negatives (missed patterns)

## Applications

### 1. DJ Mix Analysis
- Track boundary detection
- Transition type classification
- Mix structure analysis
- Energy flow visualization

### 2. Production Analysis
- Sound design deconstruction
- Arrangement analysis
- Effect chain detection
- Reference track comparison

### 3. Music Information Retrieval
- Similar track identification
- Style classification
- Groove pattern matching
- VIP/Dubplate detection

## Visualization and Reporting

The framework includes comprehensive visualization tools for:
- Spectral analysis results
- Component relationships
- Groove patterns
- Transition points
- Similarity matrices
- Analysis confidence scores

## Future Directions

1. **Enhanced Neural Processing**
   - Integration of deep learning models
   - Real-time processing capabilities
   - Adaptive threshold optimization

2. **Extended Analysis Capabilities**
   - Additional genre support
   - Extended effect detection
   - Advanced pattern recognition
   - Further error resilience improvements

3. **Improved Visualization**
   - Interactive dashboards
   - 3D visualization options
   - Real-time visualization
   - Error diagnostics visualization

## Extended References

1. Chen, J. L., Penhune, V. B., & Zatorre, R. J. (2008). Listening to musical rhythms recruits motor regions of the brain. Cerebral Cortex, 18(12), 2844-2854.

2. Cannon, J. J., & Patel, A. D. (2020). How beat perception co-opts motor neurophysiology. Trends in Cognitive Sciences, 24(1), 51-64.

3. Fukuie, T., et al. (2022). Neural entrainment reflects temporal predictions guiding speech comprehension. Current Biology, 32(5), 1051-1067.

4. Smith, J. O. (2011). Spectral Audio Signal Processing. W3K Publishing.

5. Bello, J. P., et al. (2005). A Tutorial on Onset Detection in Music Signals. IEEE Transactions on Speech and Audio Processing.

6. Gouyon, F., & Dixon, S. (2005). A Review of Automatic Rhythm Description Systems. Computer Music Journal.

7. Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.

8. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. Proceedings of the 14th Python in Science Conference.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{heihachi2024,
  title = {Heihachi: Neural Processing of Electronic Music},
  author = {[Author Names]},
  year = {2024},
  url = {https://github.com/[username]/heihachi}
}
```

### Extended System Architecture

```mermaid
graph TD
    A[Audio Input] --> B[Feature Extraction]
    B --> C[Analysis Pipeline]
    C --> D[Results Generation]

    subgraph "Feature Extraction"
        B1[Spectral] --> B2[Temporal]
        B2 --> B3[Rhythmic]
        B3 --> B4[Component]
    end

    subgraph "Analysis Pipeline"
        C1[Pattern Recognition]
        C2[Similarity Analysis]
        C3[Structure Analysis]
        C4[Effect Analysis]
    end

    subgraph "Results Generation"
        D1[Visualization]
        D2[Storage]
        D3[Export]
    end
```

### Robust Audio Processing Pipeline

```mermaid
graph TD
    A[Audio Input] --> B[Input Validation]
    B -->|Valid| C[Chunked Loading]
    B -->|Invalid| E[Error Handling]
    C --> D[Signal Processing]
    D --> F[Result Generation]
    E --> G[Recovery Mechanisms]
    G -->|Recoverable| C
    G -->|Not Recoverable| H[Error Reporting]
    
    subgraph "Input Validation"
        B1[File Exists Check]
        B2[Format Validation]
        B3[Sample Rate Check]
        B4[Empty Signal Check]
    end

    subgraph "Signal Processing"
        D1[Window Function Application]
        D2[Spectral Analysis with Hann Window]
        D3[Memory-Optimized Processing]
        D4[Null Value Handling]
    end
```

### Neurofunk Component Interaction

```mermaid
graph LR
    A[Audio Stream] --> B[Component Separation]
    B --> C[Feature Analysis]
    C --> D[Pattern Recognition]
    
    subgraph "Component Separation"
        B1[Sub Bass]
        B2[Reese Bass]
        B3[Drums]
        B4[Effects]
    end

    subgraph "Feature Analysis"
        C1[Spectral Features]
        C2[Temporal Features]
        C3[Modulation Features]
    end

    subgraph "Pattern Recognition"
        D1[Rhythmic Patterns]
        D2[Effect Patterns]
        D3[Bass Patterns]
    end
```

### Processing Pipeline Details

```mermaid
graph TD
    A[Input] --> B[Preprocessing]
    B --> C[Analysis]
    C --> D[Results]

    subgraph "Preprocessing"
        B1[Normalization]
        B2[Segmentation]
        B3[Enhancement]
    end

    subgraph "Analysis"
        C1[Feature Extraction]
        C2[Pattern Analysis]
        C3[Component Analysis]
    end

    subgraph "Results"
        D1[Metrics]
        D2[Visualizations]
        D3[Reports]
    end
```

### Technical Implementation Details

#### Bass Sound Design Analysis
```python
class BassAnalyzer:
    """Advanced bass analysis system."""
    
    def analyze_reese(self, audio: np.ndarray) -> Dict:
        """
        Analyze Reese bass characteristics.
        
        Parameters:
            audio (np.ndarray): Input audio signal
            
        Returns:
            Dict containing:
            - modulation_depth: Float
            - phase_correlation: Float
            - harmonic_content: np.ndarray
            - stereo_width: Float
        """
        pass

    def analyze_sub(self, audio: np.ndarray) -> Dict:
        """
        Analyze sub bass characteristics.
        
        Parameters:
            audio (np.ndarray): Input audio signal
            
        Returns:
            Dict containing:
            - fundamental_freq: Float
            - energy: Float
            - phase_alignment: Float
            - distortion: Float
        """
        pass
```

#### Effect Chain Analysis
```python
class EffectChainAnalyzer:
    """Advanced effect chain analysis."""
    
    def detect_chain(self, audio: np.ndarray) -> List[Dict]:
        """
        Detect processing chain order.
        
        Parameters:
            audio (np.ndarray): Input audio signal
            
        Returns:
            List[Dict] containing detected effects in order:
            - effect_type: str
            - parameters: Dict
            - confidence: Float
        """
        pass

    def analyze_parallel(self, audio: np.ndarray) -> Dict:
        """
        Analyze parallel processing.
        
        Parameters:
            audio (np.ndarray): Input audio signal
            
        Returns:
            Dict containing:
            - bands: List[Dict]
            - routing: Dict
            - blend_type: str
        """
        pass
```

#### Robust Audio Processing
```python
class AudioProcessor:
    """Memory-efficient and fault-tolerant audio processing."""
    
    def load(self, file_path: str, start_time: float = 0.0, end_time: float = None) -> np.ndarray:
        """
        Load audio file with robust error handling.
        
        Parameters:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
            
        Raises:
            ValueError: If file empty or corrupted
        """
        # Validates file existence
        # Checks for empty audio
        # Handles sample rate conversion
        # Returns normalized audio data
        pass
    
    def _chunked_load(self, file_path: str, start_time: float, end_time: float) -> np.ndarray:
        """
        Memory-efficient loading for large audio files.
        
        Parameters:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            numpy.ndarray: Audio data
        """
        # Processes file in manageable chunks
        # Handles memory efficiently
        # Validates output before returning
        pass
```
