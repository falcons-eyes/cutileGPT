# SPDX-License-Identifier: Apache-2.0
"""
GPT Model Configuration
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT models."""
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Predefined configurations
    @classmethod
    def gpt_nano(cls) -> 'GPTConfig':
        """Smallest config for testing: 3 layers, 48 dims, 3 heads"""
        return cls(n_layer=3, n_head=3, n_embd=48, block_size=128)

    @classmethod
    def gpt_micro(cls) -> 'GPTConfig':
        """Micro config: 4 layers, 128 dims, 4 heads"""
        return cls(n_layer=4, n_head=4, n_embd=128, block_size=256)

    @classmethod
    def gpt_mini(cls) -> 'GPTConfig':
        """Mini config: 6 layers, 192 dims, 6 heads"""
        return cls(n_layer=6, n_head=6, n_embd=192, block_size=256)

    @classmethod
    def gpt2(cls) -> 'GPTConfig':
        """GPT-2 (124M params): 12 layers, 768 dims, 12 heads"""
        return cls(n_layer=12, n_head=12, n_embd=768, block_size=1024)

    @classmethod
    def gpt2_medium(cls) -> 'GPTConfig':
        """GPT-2 Medium (345M params)"""
        return cls(n_layer=24, n_head=16, n_embd=1024, block_size=1024)

    @classmethod
    def gpt2_large(cls) -> 'GPTConfig':
        """GPT-2 Large (774M params)"""
        return cls(n_layer=36, n_head=20, n_embd=1280, block_size=1024)

    @classmethod
    def gpt2_xl(cls) -> 'GPTConfig':
        """GPT-2 XL (1558M params)"""
        return cls(n_layer=48, n_head=25, n_embd=1600, block_size=1024)

    # Tile-optimized configs (power of 2 dimensions)
    @classmethod
    def gpt_tile_small(cls) -> 'GPTConfig':
        """Tile-optimized small: 4 layers, 64 dims, 4 heads"""
        return cls(n_layer=4, n_head=4, n_embd=64, block_size=128)

    @classmethod
    def gpt_tile_medium(cls) -> 'GPTConfig':
        """Tile-optimized medium: 6 layers, 128 dims, 4 heads"""
        return cls(n_layer=6, n_head=4, n_embd=128, block_size=256)

    @classmethod
    def gpt_tile_large(cls) -> 'GPTConfig':
        """Tile-optimized large: 8 layers, 256 dims, 8 heads"""
        return cls(n_layer=8, n_head=8, n_embd=256, block_size=512)

    @classmethod
    def from_name(cls, name: str) -> 'GPTConfig':
        """Create config from name string."""
        configs = {
            'nano': cls.gpt_nano,
            'micro': cls.gpt_micro,
            'mini': cls.gpt_mini,
            'gpt2': cls.gpt2,
            'gpt2-medium': cls.gpt2_medium,
            'gpt2-large': cls.gpt2_large,
            'gpt2-xl': cls.gpt2_xl,
            'tile-small': cls.gpt_tile_small,
            'tile-medium': cls.gpt_tile_medium,
            'tile-large': cls.gpt_tile_large,
        }
        if name not in configs:
            raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
        return configs[name]()
