import tempfile
import unittest
from pathlib import Path
import numpy as np
from ml.config import load_config
from ml.datasets import SiWAdapter
from ml.preprocessing.rppg import extract_chrom_signal

class FoundationTests(unittest.TestCase):
    def test_config_loads(self): self.assertIn("datasets", load_config())
    def test_missing_dataset_is_clear(self): self.assertIn("not available", SiWAdapter("definitely-missing").availability_message().lower())
    def test_chrom_signal_is_finite(self):
        signal = extract_chrom_signal(np.random.default_rng(1).integers(0, 255, (8, 32, 32, 3), dtype=np.uint8))
        self.assertEqual(signal.shape, (8,)); self.assertTrue(np.isfinite(signal).all())

if __name__ == "__main__": unittest.main()
