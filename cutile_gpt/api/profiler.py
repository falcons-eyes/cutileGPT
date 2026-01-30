# SPDX-License-Identifier: Apache-2.0
"""
Data Profiler for Tile Programming

Automatic data analysis and profiling for optimal tile configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DataProfile:
    """
    Auto-detected data characteristics.

    Tile Programming Philosophy: Know your data before processing.
    """
    name: str = ""
    source: str = ""  # "huggingface", "local", "custom"

    # Shape analysis
    sample_count: int = 0
    sequence_lengths: List[int] = field(default_factory=list)
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    min_sequence_length: int = 0

    # Vocabulary analysis
    vocab_size: int = 0
    unique_tokens: int = 0
    token_distribution: Dict[int, int] = field(default_factory=dict)

    # Recommended tile configuration
    recommended_tile_m: int = 64
    recommended_tile_n: int = 64
    recommended_batch_size: int = 1

    # Memory estimation
    estimated_memory_mb: float = 0.0

    def __repr__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    DATA PROFILE                              ║
╠══════════════════════════════════════════════════════════════╣
║ Name: {self.name:<54} ║
║ Source: {self.source:<52} ║
╠══════════════════════════════════════════════════════════════╣
║ Samples: {self.sample_count:<51} ║
║ Sequence Length: min={self.min_sequence_length}, max={self.max_sequence_length}, avg={self.avg_sequence_length:.1f}
║ Vocabulary Size: {self.vocab_size:<43} ║
║ Unique Tokens Used: {self.unique_tokens:<40} ║
╠══════════════════════════════════════════════════════════════╣
║ RECOMMENDED TILE CONFIG (auto-detected):                     ║
║   tile_m: {self.recommended_tile_m:<50} ║
║   tile_n: {self.recommended_tile_n:<50} ║
║   batch_size: {self.recommended_batch_size:<47} ║
║ Est. Memory: {self.estimated_memory_mb:.1f} MB{' ':<40}║
╚══════════════════════════════════════════════════════════════╝
"""


class DataAnalyzer:
    """
    Automatic data analysis and profiling.

    Tile Programming Philosophy: Understand data characteristics
    to optimize tile configurations automatically.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def analyze(
        self,
        data: Any,
        name: str = "dataset",
        max_samples: int = 1000
    ) -> DataProfile:
        """
        Analyze data and return a profile with recommendations.

        Args:
            data: Dataset or list of samples
            name: Name for the profile
            max_samples: Maximum samples to analyze

        Returns:
            DataProfile with auto-detected characteristics
        """
        profile = DataProfile(name=name)

        # Detect data source and format
        if hasattr(data, '__class__') and 'Dataset' in data.__class__.__name__:
            profile.source = "huggingface"
            samples = self._extract_hf_samples(data, max_samples)
        elif isinstance(data, list):
            profile.source = "list"
            samples = data[:max_samples]
        elif isinstance(data, np.ndarray):
            profile.source = "array"
            samples = [data] if data.ndim == 1 else list(data)
        else:
            profile.source = "custom"
            samples = list(data)[:max_samples]

        # Tokenize if needed
        tokenized_samples = self._ensure_tokenized(samples)

        # Analyze sequences
        profile.sample_count = len(tokenized_samples)
        profile.sequence_lengths = [len(s) for s in tokenized_samples if len(s) > 0]

        if profile.sequence_lengths:
            profile.avg_sequence_length = np.mean(profile.sequence_lengths)
            profile.max_sequence_length = max(profile.sequence_lengths)
            profile.min_sequence_length = min(profile.sequence_lengths)

        # Analyze vocabulary
        all_tokens = []
        for s in tokenized_samples:
            all_tokens.extend(s if isinstance(s, list) else s.tolist())
        profile.unique_tokens = len(set(all_tokens))

        if self.tokenizer:
            profile.vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
        else:
            profile.vocab_size = max(all_tokens) + 1 if all_tokens else 50257

        # Compute recommendations
        profile = self._compute_recommendations(profile)

        return profile

    def _extract_hf_samples(self, dataset, max_samples: int) -> List:
        """Extract samples from HuggingFace dataset."""
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            if 'text' in item:
                samples.append(item['text'])
            elif 'content' in item:
                samples.append(item['content'])
            elif 'input_ids' in item:
                samples.append(item['input_ids'])
            else:
                for v in item.values():
                    if isinstance(v, str):
                        samples.append(v)
                        break
        return samples

    def _ensure_tokenized(self, samples: List) -> List:
        """Ensure all samples are tokenized."""
        tokenized = []
        for s in samples:
            if isinstance(s, str):
                if self.tokenizer:
                    tokens = self.tokenizer.encode(s)
                else:
                    tokens = [ord(c) % 50257 for c in s]
                tokenized.append(tokens)
            elif isinstance(s, (list, np.ndarray)):
                tokenized.append(list(s) if isinstance(s, np.ndarray) else s)
            else:
                tokenized.append([int(s)])
        return tokenized

    def _compute_recommendations(self, profile: DataProfile) -> DataProfile:
        """Compute optimal tile configuration based on data characteristics."""
        avg_seq = profile.avg_sequence_length

        if avg_seq <= 128:
            profile.recommended_tile_m = 32
            profile.recommended_tile_n = 32
        elif avg_seq <= 512:
            profile.recommended_tile_m = 64
            profile.recommended_tile_n = 64
        else:
            profile.recommended_tile_m = 128
            profile.recommended_tile_n = 128

        # Batch size based on memory constraints
        estimated_per_sample_mb = (profile.max_sequence_length * 768 * 4) / (1024 * 1024)
        available_mb = 8000  # Assume 8GB GPU
        profile.recommended_batch_size = max(1, int(available_mb / (estimated_per_sample_mb * 50)))
        profile.recommended_batch_size = min(profile.recommended_batch_size, 32)

        # Memory estimation
        profile.estimated_memory_mb = estimated_per_sample_mb * profile.recommended_batch_size * 10

        return profile
