# Heihachi Performance Optimization

This document provides an overview of the performance optimization features implemented in the Heihachi audio analysis framework.

## Table of Contents

1. [Overview](#overview)
2. [Performance Configuration](#performance-configuration)
3. [Profiling](#profiling)
4. [Memory Optimization](#memory-optimization)
5. [Parallel Processing](#parallel-processing)
6. [Caching](#caching)
7. [Visualization Optimization](#visualization-optimization)
8. [Result Comparison](#result-comparison)
9. [Batch Processing](#batch-processing)
10. [Interactive Mode](#interactive-mode)
11. [Usage Examples](#usage-examples)

## Overview

The Heihachi audio analysis framework has been optimized for performance in several key areas:

- **Profiling**: Monitor code execution time and identify bottlenecks
- **Memory optimization**: Efficient memory usage and garbage collection
- **Parallel processing**: Process multiple audio files concurrently
- **Caching**: Store and reuse intermediate results
- **Visualization optimization**: Handle large audio files efficiently
- **Result comparison**: Compare analysis results from multiple files
- **Batch processing**: Process multiple files with different configurations
- **Interactive mode**: Real-time audio analysis and visualization

## Performance Configuration

The framework uses a dedicated performance configuration file (`configs/performance.yaml`) that allows you to tune various performance parameters:

```yaml
# Processing settings
processing:
  num_workers: 4                 # Number of parallel workers
  batch_size: 16                 # Batch size for batch processing
  chunk_size: 4096               # Audio chunk size in samples
  max_file_size_mb: 100          # Maximum file size to process
  memory_limit_mb: 1024          # Memory limit for processing

# Audio processing
audio:
  sample_rate: 44100             # Sample rate for processing
  normalize: true                # Normalize audio before processing
  n_fft: 2048                    # FFT size
  hop_length: 512                # Hop length for spectrogram

# Cache settings
cache:
  enabled: true                  # Enable caching
  memory_cache_mb: 512           # Memory cache size
  disk_cache_mb: 1024            # Disk cache size
  compression_level: 5           # Compression level (1-9)
  ttl_seconds: 3600              # Time-to-live for cache entries

# Storage settings
storage:
  compression_enabled: true      # Enable compression for stored files
  compression_format: "gzip"     # Compression format (gzip, bz2, etc.)
  use_mmap: true                 # Use memory-mapped files for large datasets

# GPU acceleration
gpu:
  enabled: false                 # Enable GPU acceleration
  memory_fraction: 0.8           # Fraction of GPU memory to use
  mixed_precision: true          # Use mixed precision for GPU calculations

# Resource management
resource_management:
  max_unused_time: 300           # Maximum time to keep unused resources (seconds)
  memory_threshold: 0.8          # Memory threshold for resource cleanup

# Visualization
visualization:
  dpi: 100                       # DPI for output visualizations
  max_points: 10000              # Maximum number of points for time series

# Logging and monitoring
logging:
  level: "INFO"                  # Logging level
  memory_monitoring: true        # Enable memory usage monitoring

# Advanced options
advanced:
  feature_extraction_threads: 2  # Threads for feature extraction
  memory_profiling: false        # Enable detailed memory profiling
```

## Profiling

The `Profiler` utility (`src/utils/profiling.py`) provides performance monitoring capabilities:

- Function execution time tracking
- Bottleneck identification
- Detailed performance reports
- Timeline visualization

Usage:

```python
from src.utils.profiling import Profiler

# Initialize profiler
profiler = Profiler(output_dir="./profiling_results")

# Start profiling
profiler.start()

# Your code here...

# Stop profiling and generate report
profiler.stop()
```

## Memory Optimization

Memory optimization includes:

- Efficient memory allocation and deallocation
- Garbage collection optimization
- Memory usage monitoring
- Resource management based on memory pressure

The `MemoryMonitor` class (`src/utils/logging_utils.py`) tracks memory usage throughout execution.

## Parallel Processing

Parallel processing is implemented in the `Pipeline` class and `BatchProcessor`:

- Configurable number of workers
- Process pool for CPU-bound tasks
- Thread pool for I/O-bound tasks
- Progress tracking for long-running operations

## Caching

The framework implements multiple caching strategies:

- `AudioCache` for caching audio data
- `IntermediateResultCache` for caching processing results
- Memory and disk caching options
- Cache statistics and management

## Visualization Optimization

The `VisualizationOptimizer` class (`src/utils/visualization_optimization.py`) provides optimized visualization generation:

- Data downsampling for large waveforms
- Optimized spectrogram rendering
- Batch visualization generation
- Memory-efficient plotting

## Result Comparison

The `ResultComparison` class (`src/utils/comparison.py`) allows comparing analysis results from multiple files:

- Statistical summaries of metrics
- Outlier detection
- Visualization of differences
- Correlation analysis
- HTML, JSON, and CSV report generation

## Batch Processing

The `BatchProcessor` class (`src/core/batch_processing.py`) enables processing multiple files:

- Directory traversal
- Multiple configuration support
- Resumable processing
- Progress tracking and reporting
- Fault tolerance

## Interactive Mode

The interactive mode (`src/commands/interactive.py`) provides a command-line interface for:

- Real-time audio analysis
- Visualization generation
- Result comparison
- Batch processing control

## Usage Examples

### Basic Processing

```bash
# Process a single file
python -m src.main --input path/to/file.wav --output-dir ./results

# Process a directory
python -m src.main --input path/to/directory --output-dir ./results
```

### Batch Processing

```bash
# Process a directory with batch processing
python -m src.main process --input path/to/directory --output-dir ./results --config ./configs/default.yaml --performance-config ./configs/performance.yaml

# Resume interrupted batch processing
python -m src.main process --input path/to/directory --output-dir ./results --resume
```

### Result Comparison

```bash
# Compare analysis results
python -m src.main compare --input-dir ./results --output-dir ./comparison_results --format html

# Compare specific files with custom metrics
python -m src.main compare --files ./results/file1.json ./results/file2.json --metrics bpm key quality_overall --output-dir ./comparison_results
```

### Interactive Mode

```bash
# Start interactive mode
python -m src.main interactive --output-dir ./interactive_results

# Start with custom configuration
python -m src.main interactive --config ./configs/custom.yaml --debug
```

## Performance Tuning Tips

1. **Adjust worker count**: Set `num_workers` based on your CPU core count
2. **Memory management**: Set appropriate `memory_limit_mb` based on your system
3. **Cache tuning**: Enable caching for repeated operations on the same files
4. **Chunk size optimization**: Adjust `chunk_size` based on your audio file characteristics
5. **GPU acceleration**: Enable GPU for spectral analysis on supported systems
6. **Compression**: Use compression for large datasets to reduce I/O bottlenecks
7. **Visualization optimization**: Reduce DPI and max_points for faster visualization

For more information, refer to the source code documentation in the respective modules. 