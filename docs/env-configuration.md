# Environment Configuration for Heihachi

This guide explains how to configure environment variables and use `.env` files for managing API keys and other sensitive information in the Heihachi audio analysis framework.

## Using `.env` Files

Heihachi supports loading environment variables from a `.env` file, which is a convenient way to manage API keys and other configuration options without hardcoding them in your code or exposing them in command-line arguments.

### Creating a `.env` File

1. Create a file named `.env` in the root directory of your Heihachi installation
2. Add your API keys and other configuration variables in the format `KEY=VALUE`
3. Make sure to keep this file secure and not commit it to version control

Example `.env` file content:

```
# Hugging Face API key for accessing models
HF_API_TOKEN=your_huggingface_api_key_here

# System configuration
HEIHACHI_CACHE_DIR=./cache
GPU_MEMORY_FRACTION=0.8
```

### Supported Environment Variables

| Variable Name | Alternative Names | Description |
|---------------|-------------------|-------------|
| `HF_API_TOKEN` | `HUGGINGFACE_API_KEY`, `HF_TOKEN` | Hugging Face API token for accessing gated models |
| `HEIHACHI_CACHE_DIR` | | Directory for caching models and processed data |
| `GPU_MEMORY_FRACTION` | | Fraction of GPU memory to use (0.0-1.0) |
| `MAX_BATCH_SIZE` | | Maximum batch size for processing |

## Using Environment Variables with Command-Line

When using the command-line interface, you can still provide API keys as arguments, which will take precedence over the `.env` file:

```bash
python -m src.main hf extract path/to/audio.mp3 --api-key YOUR_API_KEY
```

However, using a `.env` file is recommended for security and convenience.

## Installing Required Packages

For optimal `.env` file support, it's recommended to install the `python-dotenv` package:

```bash
pip install python-dotenv
```

Heihachi will automatically use this package if it's available, or fall back to a basic implementation if it's not installed.

## Troubleshooting

If your API key isn't being recognized:

1. Make sure your `.env` file is in the correct location (project root directory)
2. Check that the variable name is correct (`HF_API_TOKEN` is preferred)
3. Verify the API key is valid and has the necessary permissions
4. Try setting the environment variable directly in your shell before running the command

For debugging environment variable loading, you can run with the `--debug` flag:

```bash
python -m src.main --debug hf extract path/to/audio.mp3
``` 