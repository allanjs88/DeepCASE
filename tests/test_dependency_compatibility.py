import io
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepcase.context_builder import ContextBuilder
from deepcase.module import DeepCASE
from deepcase.preprocessing import Preprocessor


class DependencyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        data = pd.DataFrame({
            'timestamp': range(24),
            'event': [0, 1, 2] * 8,
            'machine': ['a'] * 24,
        })
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self.X, self.y, self.labels, self.mapping = Preprocessor(
                length=3, timeout=86400,
            ).sequence(data)

    def test_train_predict_and_checkpoint_round_trip(self):
        model = DeepCASE(
            features=len(self.mapping), max_length=3, hidden_size=8,
            min_samples=1, threshold=0,
        )
        targets = self.y.reshape(-1, 1)
        model.fit(
            self.X, targets, np.zeros(24), epochs=1, batch_size=8,
            iterations=2, verbose=False,
        )
        expected = model.predict(self.X, targets, iterations=2, verbose=False)
        np.testing.assert_array_equal(expected, np.zeros(24))
        checkpoint = io.BytesIO()
        model.save(checkpoint)
        checkpoint.seek(0)
        restored = DeepCASE.load(checkpoint, device='cpu')
        np.testing.assert_array_equal(
            restored.predict(self.X, targets, iterations=2, verbose=False),
            expected,
        )
        checkpoint = io.BytesIO()
        model.context_builder.save(checkpoint)
        checkpoint.seek(0)
        builder = ContextBuilder.load(checkpoint, device='cpu')
        for key, value in model.context_builder.state_dict().items():
            torch.testing.assert_close(builder.state_dict()[key], value)

    def test_cli_loads_saved_numpy_labels_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'sequences.pt'
            torch.save({
                'context': self.X, 'events': self.y,
                'labels': self.labels, 'mapping': self.mapping,
            }, filename)
            result = subprocess.run(
                [sys.executable, '-m', 'deepcase', 'sequence',
                 '--load-sequences', str(filename), '--silent'],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
