import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example.reporting.experiment import generate_graphs
from example.reporting.utils import confusion_report
from deepcase.utils import confusion_report as original_report


class ReportingTests(unittest.TestCase):
    def test_unscored_attacks_remain_separate(self):
        data = pd.DataFrame({"label": [0, 0, 1, 1], "prediction": [0, 1, -1, 0]})
        with tempfile.TemporaryDirectory() as directory:
            generate_graphs(data, {"loss_history": [0.0], "timing": {"training_seconds": 0.0}},
                            directory, "test")
            matrix = pd.read_csv(Path(directory) / "confusion_matrix.csv", index_col=0)
            self.assertEqual(matrix.loc["Attack", "Unscored"], 1)
            self.assertEqual(matrix.loc["Attack", "Benign"], 1)
            self.assertEqual(matrix.to_numpy().sum(), 4)
            for name in ["confusion_matrix.png", "training_loss.png", "processing_time.png"]:
                self.assertGreater((Path(directory) / name).stat().st_size, 0)

    def test_copied_report_matches_original_and_imports_still_work(self):
        options = dict(labels=[0, 1, -1], target_names=["Benign", "Attack", "Unscored"],
                       skip_y={"Unscored"})
        self.assertEqual(confusion_report([0, 1], [0, -1], **options),
                         original_report([0, 1], [0, -1], **options))
        from example.plotting import ResultsPlotter
        from example.reporting.plotting import ResultsPlotter as moved
        self.assertIs(ResultsPlotter, moved)
