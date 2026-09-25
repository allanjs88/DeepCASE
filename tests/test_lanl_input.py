import tempfile
import unittest
from pathlib import Path
import pandas as pd
from example.example_lanl import make_lanl_frame, partition_index, PAPER_SPLIT_TIME


class LanlInputTests(unittest.TestCase):
    def test_filter_metadata_and_exact_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            auth, red = Path(directory) / 'auth.txt', Path(directory) / 'red.txt'
            rows = [
                '150000,C1$@DOM1,C1$@DOM1,C1,C2,Kerberos,Network,LogOn,Success',
                '10,SYSTEM,SYSTEM,C2,C3,Negotiate,Network,LogOn,Fail',
                '11,U,U,C2,C2,NTLM,Network,LogOn,Success',
                '12,U,U,C2,C3,NTLM,Network,LogOff,Success',
                '13,U,U,?,C3,NTLM,Network,LogOn,Success',
                '150000,C1$@DOM1,C1$@DOM1,C1,C2,Kerberos,Network,LogOn,Success',
                '150000,V,V,C1,C2,NTLM,Network,LogOn,Success',
            ]
            auth.write_text('\n'.join(rows) + '\n')
            red.write_text('150000,C1$@DOM1,C1,C2\n')
            frame = make_lanl_frame(auth, red)
            self.assertEqual(frame.event_id.tolist(), [1, 0, 5, 6])
            self.assertEqual(frame.label.tolist(), [0, 1, 1, 0])
            self.assertEqual(frame.destination_computer.tolist(), ['C3', 'C2', 'C2', 'C2'])
            self.assertEqual(len(make_lanl_frame(auth, red, auth_scope='all')), 7)
            self.assertEqual(partition_index(frame), 1)

    def test_cutoff(self):
        frame = pd.DataFrame({'timestamp': [1, PAPER_SPLIT_TIME - 1, PAPER_SPLIT_TIME,
                                           PAPER_SPLIT_TIME], 'label': [0, 0, 1, 0]})
        self.assertEqual(partition_index(frame), 2)
        with self.assertRaises(ValueError):
            partition_index(frame.iloc[:2])
