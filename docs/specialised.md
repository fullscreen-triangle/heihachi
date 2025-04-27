# Specialized AI Models for Heihachi

This document catalogs AI models from Hugging Face that are valuable for audio analysis in the Heihachi framework, organized by task category.

## Audio Analysis Models

### Core Feature Extraction

| Task | HF Model | Description | Implementation Priority |
|------|----------|-------------|------------------------|
| Generic spectral + temporal embeddings | [microsoft/BEATs](https://huggingface.co/microsoft/BEATs) family (base & large) | Bidirectional ViT-style encoder trained with acoustic tokenisers; gives 768-d latent at ~20 ms hop. Fine-tune for segment clustering or effect-chain detection. | High |
| Robust speech & non-speech features | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (encoder only) | Trained on >5M hours; encoder provides 1280-d features tracking energy, voicing & language. | High |

### Multimodal & Similarity

| Task | HF Model | Description | Implementation Priority |
|------|----------|-------------|------------------------|
| Multimodal similarity / tagging | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | Query with free-text ("syncopated jungle drum break") and compute cosine similarity on 512-d embeddings. | Medium |
| Zero-shot tag & prompt embedding | [UniMus/OpenJMLA](https://huggingface.co/UniMus/OpenJMLA) | Score arbitrary tag strings for effect-chain heuristics. | Medium |
| Large-scale music tagging | HTSAT checkpoints inside CLAP | Already trained on AudioSet; fine-tune last MLP layer for drum & bass sub-genres. | Medium |

## Rhythm & Beat Analysis

| Task | HF Model | Description | Implementation Priority |
|------|----------|-------------|------------------------|
| Beat / down-beat tracking | [Beat-Transformer](https://huggingface.co/nicolaus625/cmi) | Dilated self-attention encoder with F-measure ~0.86. Works on 128×128 mel patches. | High |
| Low-latency streaming beat-tracking | [BEAST](https://github.com/beats-team/beast) | 50 ms latency, causal attention; ideal for real-time DJ analysis. | Medium |
| Drum-onset / kit piece ID | [DunnBC22/wav2vec2-base-Drum_Kit_Sounds](https://huggingface.co/DunnBC22/wav2vec2-base-Drum_Kit_Sounds) | Fine-tuned on kick/snare/tom/overhead labels; helps with pattern-matching. | Medium |

## Audio Separation & Component Analysis

| Task | HF Model | Description | Implementation Priority |
|------|----------|-------------|------------------------|
| Stem isolation (sub-bass, drums, etc.) | [Demucs v4](https://huggingface.co/spaces/abidlabs/music-separation) | Returns 4-stem or 6-stem tensors for component-level analysis. | High |

## Future Extensions

| Task | HF Starting Point | Implementation Notes |
|------|-------------------|---------------------|
| Audio captioning | [slseanwu/beats-conformer-bart-audio-captioner](https://huggingface.co/slseanwu/beats-conformer-bart-audio-captioner) | Produces textual descriptions per segment. |
| Similarity retrieval UI | CLAP embeddings + FAISS | Index embeddings and expose nearest-neighbor search. |

## Implementation Plan

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

## Usage Examples

```python
# Example usage of BEATs model for feature extraction
from heihachi.huggingface import FeatureExtractor

extractor = FeatureExtractor(model="microsoft/BEATs-base")
features = extractor.extract(audio_path="track.mp3")

# Example of stem separation with Demucs
from heihachi.huggingface import StemSeparator

separator = StemSeparator()
stems = separator.separate(audio_path="track.mp3")
bass_stem = stems["bass"]
drums_stem = stems["drums"]
```