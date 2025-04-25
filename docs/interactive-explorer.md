# Interactive Explorer and Progress Utilities

This document provides information on how to use the interactive explorer for analyzing audio results and the progress tracking utilities for long-running operations.

## Table of Contents

1. [Interactive Explorer](#interactive-explorer)
   - [Command-Line Explorer](#command-line-explorer)
   - [Web UI](#web-ui)
   - [Commands Reference](#commands-reference)
2. [Progress Utilities](#progress-utilities)
   - [Simple Progress Bar](#simple-progress-bar)
   - [Progress Context](#progress-context)
   - [Progress Manager](#progress-manager)
   - [CLI Progress Bar](#cli-progress-bar)
3. [Batch Processing](#batch-processing)
   - [Batch Processing CLI](#batch-processing-cli)
   - [Batch Configuration](#batch-configuration)
4. [Demonstration](#demonstration)

## Interactive Explorer

Heihachi provides an interactive explorer for analyzing and visualizing audio analysis results. The explorer is available in two forms:

1. **Command-Line Interface (CLI)**: A text-based interactive explorer
2. **Web UI**: A browser-based visualization and exploration tool

### Command-Line Explorer

The CLI explorer provides an interactive command-line interface for exploring audio analysis results, with commands for:

- Listing available result files
- Opening and examining result files
- Visualizing features and waveforms
- Comparing results across multiple files
- Exporting data to various formats

#### Starting the CLI Explorer

To start the CLI explorer, use the interactive_cli.py module:

```bash
python -m src.cli.interactive_cli --results-dir /path/to/results
```

#### Options

- `--results-dir`: Directory containing analysis results (default: `results`)
- `--vis-dir`: Directory to save visualizations (default: `visualizations`)
- `--log-level`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)

### Web UI

The Web UI provides a graphical interface for exploring and visualizing audio analysis results in a web browser.

#### Starting the Web UI

```bash
python -m src.cli.interactive_cli --web --result-file /path/to/result.json
```

#### Options

- `--web`: Start web UI instead of CLI explorer
- `--host`: Host to bind the web server to (default: `localhost`)
- `--port`: Port to bind the web server to (default: `5000`)
- `--result-file`: Specific result file to load in web UI
- `--no-browser`: Don't automatically open browser

### Commands Reference

The CLI explorer supports the following commands:

| Command | Alias | Description |
|---------|-------|-------------|
| `list [pattern]` | `ls`, `l` | List available analysis result files |
| `open INDEX/FILENAME` | `o` | Open an analysis result file |
| `summary` | `s` | Show summary of the current analysis result |
| `info FEATURE_NAME` | `i` | Show detailed information about a specific feature |
| `plot TYPE [FEATURE_NAME] [--save FILENAME]` | `p` | Plot a feature or visualization |
| `compare FEATURE_NAME [file1 file2 ...]` | - | Compare a feature across multiple results |
| `distribution FEATURE_NAME [file1 file2 ...]` | - | Plot distribution of a feature |
| `export FORMAT [FILENAME]` | `e` | Export the current result to another format |
| `refresh` | - | Refresh the list of available result files |
| `help` | `?` | Show help for commands |
| `exit`, `quit` | `q` | Exit the explorer |

#### Plot Types

The `plot` command supports the following types:

- `waveform`: Plot audio waveform
- `spectrogram`: Plot audio spectrogram
- `feature FEATURE_NAME`: Plot a specific feature

#### Export Formats

The `export` command supports the following formats:

- `csv`: Export features to CSV
- `json`: Pretty format the JSON
- `markdown`: Export summary as Markdown

## Progress Utilities

Heihachi provides utilities for tracking progress of long-running operations. These utilities are defined in `src/utils/progress.py` and can be used to provide visual feedback during analysis, batch processing, and other time-consuming tasks.

### Simple Progress Bar

The `SimpleProgress` class provides a basic progress bar for simple console output:

```python
from src.utils.progress import SimpleProgress

# Create a simple progress bar
progress = SimpleProgress(total=100, desc="Processing", width=40)

# Update progress
for i in range(100):
    progress.update(1)  # Increment by 1
    # Do some work...

# Ensure it's finished
progress.finish()
```

### Progress Context

The `progress_context` context manager provides a more sophisticated progress bar using the `rich` library:

```python
from src.utils.progress import progress_context

# Use as a context manager
with progress_context(description="Processing data", total=100) as update_progress:
    # Initial step
    update_progress(10, "Loading data")
    
    # Perform work...
    
    # Update progress with status
    update_progress(50, "Analyzing")
    
    # More work...
    
    # Final update
    update_progress(40, "Finalizing")
```

### Progress Manager

The `ProgressManager` class allows tracking multiple operations simultaneously:

```python
from src.utils.progress import ProgressManager

# Create a progress manager
with ProgressManager() as progress:
    # Start tracking operations
    progress.start_operation("load", "Loading data", total=100)
    progress.start_operation("analyze", "Analyzing audio", total=100)
    
    # Update first operation
    progress.update("load", 50, "Loading file 1/2")
    # Do some work...
    progress.update("load", 50, "Loading file 2/2")
    
    # Update second operation
    progress.update("analyze", 30, "Extracting features")
    # Do some work...
    progress.update("analyze", 70, "Computing metrics")
    
    # Mark operations as complete
    progress.complete("load", "Data loaded successfully")
    progress.complete("analyze", "Analysis complete")
    
    # If an error occurs
    # progress.error("operation_id", "Error message")
```

### CLI Progress Bar

The `cli_progress_bar` function provides a wrapper around `tqdm` for simple progress bars:

```python
from src.utils.progress import cli_progress_bar

# Use with an iterable
for item in cli_progress_bar(items, desc="Processing items", unit="file"):
    # Process item...

# Use with explicit updates
with cli_progress_bar(total=100, desc="Processing") as pbar:
    # Do some work...
    pbar.update(10)
    # More work...
    pbar.update(20)
```

## Batch Processing

Heihachi includes a batch processing system for analyzing multiple audio files with different configurations.

### Batch Processing CLI

The batch processing CLI allows processing multiple files with different configurations:

```bash
python -m src.cli.batch_cli --input-dir /path/to/audio --config /path/to/config.json
```

#### Options

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

### Batch Configuration

Batch processing can also be configured through a JSON file:

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

## Demonstration

To see the interactive explorer and progress utilities in action, run the demonstration script:

```bash
python scripts/interactive_demo.py
```

This script will:

1. Check if sample audio files are available
2. Process sample files with progress indicators
3. Launch the interactive explorer or web UI

Additional options:

- `--skip-processing`: Skip sample processing and use existing results
- `--progress-demo`: Show progress indicators demonstration only

This demonstration provides a good starting point for understanding how to use the interactive explorer and progress tracking utilities in your own code. 