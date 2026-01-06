"""
Historical OCR Dataset Loader
Loads pre-cropped image-text pairs from JSONL manifest files.
Supports the traditional-chinese-historical-ocr-lo-chia-lun dataset format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image


class HistoricalOCRDataset:
    """Dataset loader for historical OCR with JSONL manifest format."""

    def __init__(self, dataset_dir: str, split: str = 'train'):
        """
        Initialize Historical OCR dataset loader.

        Args:
            dataset_dir: Root directory containing manifests/ and crops/
            split: One of 'train', 'val', 'test', or 'all'
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.samples = []
        self._load_manifest()

    def _load_manifest(self):
        """Load samples from JSONL manifest file."""
        manifest_path = self.dataset_dir / 'manifests' / f'{self.split}.jsonl'

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: PIL Image of cropped text region
                - text: Ground truth text
                - text_direction: 'ltr' or 'rtl'
                - metadata: Additional information
        """
        sample = self.samples[idx]

        # Load pre-cropped image
        crop_path = self.dataset_dir / sample['crop_path']
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop image not found: {crop_path}")

        image = Image.open(crop_path).convert('RGB')

        return {
            'image': image,
            'text': sample['text'],
            'text_direction': sample['text_direction'],
            'metadata': {
                'id': sample['id'],
                'rec_id': sample['rec_id'],
                'page_id': sample['page_id'],
                'shape_id': sample['shape_id'],
                'group_id': sample['group_id'],
                'order': sample['order'],
                'crop_path': str(crop_path),
                'source_image': sample['source_image'],
                'split': sample['split']
            }
        }

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics about the dataset
        """
        stats = {
            'total_samples': len(self.samples),
            'split': self.split,
            'records': {},
            'text_directions': {'ltr': 0, 'rtl': 0},
            'text_length_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0
            }
        }

        text_lengths = []
        for sample in self.samples:
            # Count by record
            rec_id = sample['rec_id']
            if rec_id not in stats['records']:
                stats['records'][rec_id] = 0
            stats['records'][rec_id] += 1

            # Count by text direction
            direction = sample['text_direction']
            if direction in stats['text_directions']:
                stats['text_directions'][direction] += 1

            # Text length
            text_len = len(sample['text'])
            text_lengths.append(text_len)

        if text_lengths:
            stats['text_length_stats']['min'] = min(text_lengths)
            stats['text_length_stats']['max'] = max(text_lengths)
            stats['text_length_stats']['mean'] = sum(text_lengths) / len(text_lengths)

        return stats


def test_dataset():
    """Test function to verify dataset loading."""
    import os
    dataset_dir = os.environ.get('HISTORICAL_DATASET_DIR', './dataset')

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Testing {split.upper()} split")
        print('='*60)

        dataset = HistoricalOCRDataset(dataset_dir, split=split)

        print(f"\nDataset size: {len(dataset)}")

        # Get statistics
        stats = dataset.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Text directions: {stats['text_directions']}")
        print(f"  Text length - Min: {stats['text_length_stats']['min']}, "
              f"Max: {stats['text_length_stats']['max']}, "
              f"Mean: {stats['text_length_stats']['mean']:.1f}")
        print(f"\nRecords:")
        for rec_id, count in sorted(stats['records'].items()):
            print(f"  {rec_id}: {count} samples")

        # Test loading a few samples
        print(f"\nTesting sample loading...")
        for i in range(min(2, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Text: {sample['text']}")
            print(f"  Direction: {sample['text_direction']}")
            print(f"  Image size: {sample['image'].size}")
            print(f"  ID: {sample['metadata']['id']}")


if __name__ == '__main__':
    test_dataset()
