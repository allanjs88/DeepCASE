import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "compare_lanl", Path(__file__).resolve().parents[1] / "example/compare_lanl.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ComparisonTests(unittest.TestCase):
    def test_context_excludes_ties_future_and_expired_events(self):
        frame = pd.DataFrame({"timestamp": [0, 1, 1, 2, 5],
                              "source_computer": ["A"] * 5,
                              "event": ["a", "b", "unknown", "a", "b"]})
        context, target = runner.original_sequences(frame, {"a": 2, "b": 3}, 3, 2)
        self.assertEqual(context.tolist(), [[0, 0, 0], [0, 0, 2], [0, 0, 2],
                                            [2, 3, 1], [0, 0, 0]])
        self.assertEqual(target.flatten().tolist(), [2, 3, 1, 2, 3])

    def test_abstentions_are_separate_from_scored_negatives(self):
        result = runner.metrics(np.array([1, 1, 1, 0, 0, 0]),
                                np.array([1, 0, -1, 1, 0, -1]),
                                np.array([True, True, False, True, True, False]))
        for key in ["tp", "fp", "tn_scored", "fn_scored",
                    "unscored_positive_events", "unscored_negative_events"]:
            self.assertEqual(result[key], 1)
        self.assertEqual(result["attacks_not_alerted_including_unscored"], 2)
        self.assertAlmostEqual(result["recall_all_positives"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
