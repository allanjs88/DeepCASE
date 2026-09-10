import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from example.prepare_lanl_interval import prepare
from example.example_lanl import partition_index


class LanlIntervalTests(unittest.TestCase):
    def test_complete_unsorted_interval_and_exact_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth = root / 'auth.gz'
            red = root / 'red.txt'
            # Later timestamps precede interval rows: extraction must scan to EOF.
            rows = ['99,U,U,A,B,T,L,O,Success',
                    '10,U,U,A,B,T,L,O,Success',
                    '12,U,U,A,B,T,L,O,Success',
                    '12,U,U,A,B,T,L,O,Success',
                    '12,V,V,A,B,T,L,O,Success',
                    '15,U,U,A,B,T,L,O,Success']
            with gzip.open(auth, 'wt') as stream:
                stream.write('\n'.join(rows) + '\n')
            red.write_text('10,U,A,B\n12,U,A,B\n14,U,A,B\n')
            output = root / 'interval.txt'
            report = prepare(auth, red, output, 10, 15, 12)
            self.assertEqual(report['scanned_rows'], 6)
            self.assertEqual(report['train']['rows'], 1)
            self.assertEqual(report['test']['rows'], 3)
            self.assertEqual(report['test']['positive_rows'], 2)
            self.assertEqual(report['test']['unique_matched_redteam_keys'], 1)
            self.assertEqual(report['test']['unmatched_redteam_keys'], 1)
            self.assertEqual(len(output.read_text().splitlines()), 4)
            with self.assertRaises(FileExistsError):
                prepare(auth, red, output, 10, 15, 12)

    def test_partition_keeps_timestamp_ties_together(self):
        data = pd.DataFrame({'timestamp': [10, 11, 11, 12], 'label': [1, 0, 1, 1]})
        self.assertEqual(partition_index(data, split_time=11), 1)
        self.assertEqual(partition_index(data, train_ratio=0.5), 1)
        with self.assertRaises(ValueError):
            partition_index(data, split_time=13)


if __name__ == '__main__':
    unittest.main()
