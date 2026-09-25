import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from deepcase_gnn import DeepCASEGNN
from deepcase_gnn.context_builder import ContextBuilder
from deepcase_gnn.interpreter import detection_metrics
from deepcase_gnn.preprocessing import Preprocessor, read_lanl


def events():
    return pd.DataFrame({
        "event_id": ["a", "b", "c", "d", "e", "f"],
        "timestamp": [1, 2, 2, 3, 5, 6],
        "source_computer": ["A", "B", "A", "C", "A", "NEW"],
        "destination_computer": ["B", "C", "B", "A", "B", "A"],
        "source_user": ["u", "v", "u", "v", "u", "new"],
        "event": ["ok", "fail", "ok", "ok", "novel", "ok"],
        "success": ["Success", "Fail", "Success", "Success", "Fail", "Success"],
        "label": [0, 0, 1, 0, 1, 0],
    })


class GraphPreprocessingTests(unittest.TestCase):
    def test_strict_past_direction_duplicates_window_and_unknown(self):
        frame = events()
        pre = Preprocessor(length=2, window_seconds=2).fit(frame.iloc[:4])
        data = pre.transform(frame)
        first, tied, third, fifth, last = [data[i] for i in range(len(data))]
        self.assertEqual(first.edge_index.shape, (2, 0))
        self.assertEqual(first.lengths.sum().item(), 0)
        # At timestamp 2, only A -> B at time 1 exists; both targets are excluded.
        self.assertEqual(tied.edge_index.tolist(), [[0], [1]])
        self.assertEqual(tied.y.numel(), 2)
        self.assertEqual(tied.edge_weight.tolist(), [1.0])
        self.assertEqual(third.edge_weight.tolist(), [2.0, 1.0])
        self.assertEqual(third.sequence[0].tolist(), [pre.vocabulary["ok"]] * 2)
        # At time 5, time 2 is outside [3,5); only C -> A remains.
        self.assertEqual(fifth.edge_index.tolist(), [[2], [0]])
        self.assertEqual(fifth.y.tolist(), [1])
        self.assertEqual(last.lengths[-1].item(), 0)  # unseen NEW host
        self.assertEqual(last.sequence[0, 0].item(), 1)  # historical novel token
        self.assertNotIn("novel", pre.vocabulary)
        for graph in (first, tied, third, fifth, last):
            self.assertTrue(graph.validate())

    def test_labels_current_category_and_future_do_not_change_inputs(self):
        frame = events()
        pre = Preprocessor().fit(frame)
        original = pre.transform(frame)[1]
        changed = frame.copy()
        changed["label"] = 1 - changed.label
        changed.loc[changed.timestamp >= 2, "event"] = "not-in-vocab"
        changed.loc[changed.timestamp > 2, "source_computer"] = "FUTURE"
        other = pre.transform(changed)[1]
        for key in ("sequence", "lengths", "x", "edge_index", "edge_attr", "target_index", "target_attr"):
            torch.testing.assert_close(original[key], other[key])
        self.assertFalse(torch.equal(original.y, other.y))

    def test_explicit_history_and_id_validation(self):
        frame = events()
        pre = Preprocessor().fit(frame.iloc[:4])
        full = pre.transform(frame)[3]
        partition = pre.transform(frame.iloc[4:], history=frame.iloc[:4])[0]
        for key in ("sequence", "x", "edge_index", "target_index", "target_attr"):
            torch.testing.assert_close(full[key], partition[key])
        with self.assertRaises(ValueError):
            pre.transform(frame.iloc[1:], history=frame.iloc[:2])
        with self.assertRaises(ValueError):
            pre.transform(pd.concat([frame, frame.iloc[:1]]))

    def test_lanl_preserves_physical_rows_and_exact_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            auth, red = Path(directory) / "auth.txt", Path(directory) / "red.txt"
            row = "2,U,U,A,B,Kerberos,Network,LogOn,Success\n"
            auth.write_text(row + "1,V,V,B,C,NTLM,Network,LogOn,Fail\n" + row
                            + "3,U,U,A,A,Kerberos,Network,LogOn,Success\n")
            red.write_text("2,U,A,B\n")
            frame = read_lanl(auth, red)
            self.assertEqual(frame.event_id.tolist(), ["1", "0", "2"])
            self.assertEqual(frame.label.tolist(), [0, 1, 1])
            self.assertEqual(frame.event.iloc[1], frame.event.iloc[2])
            self.assertEqual(len(read_lanl(auth, auth_scope="all")), 4)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny fixtures are much faster without a large CPU thread pool.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_batch_offsets_equivalence_and_end_to_end_gradients(self):
        pre = Preprocessor().fit(events())
        dataset = pre.transform(events())
        first, second = dataset[1], dataset[2]
        batch = Batch.from_data_list([first, second])
        torch.testing.assert_close(batch.target_index[:, first.y.numel():],
                                   second.target_index + first.num_nodes)
        model = ContextBuilder(pre.features, hidden_size=8, embedding_size=4, dropout=0)
        model.eval()
        expected = torch.cat([model(first), model(second)])
        torch.testing.assert_close(model(batch), expected)
        torch.nn.functional.cross_entropy(model(batch), batch.y - 1).backward()
        for parameters in (model.gru.parameters(), model.convs.parameters(), model.event_head.parameters()):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters if p.grad is not None), 0)
        modified = second.clone()
        modified.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.assertFalse(torch.allclose(model(second), model(modified)))

    def test_training_label_independence_calibration_and_checkpoint(self):
        frame = events()
        train, validation, test = frame.iloc[:4], frame.iloc[4:5], frame.iloc[5:]
        kwargs = dict(hidden_size=8, embedding_size=4, dropout=0, seed=7)
        model = DeepCASEGNN(**kwargs).fit(train, epochs=3, batch_size=2)
        other = DeepCASEGNN(**kwargs).fit(train.assign(label=1 - train.label), epochs=3, batch_size=2)
        self.assertEqual(model.loss_history, other.loss_history)
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        model.calibrate(validation, history=train)
        result = model.predict(test, history=frame.iloc[:5])
        self.assertEqual(result.event_id.tolist(), ["f"])
        self.assertTrue(np.isfinite(result.score).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = DeepCASEGNN.load(path)
            pd.testing.assert_frame_equal(result, loaded.predict(test, history=frame.iloc[:5]))
            self.assertEqual(model.preprocessor.vocabulary, loaded.preprocessor.vocabulary)

    def test_empty_graph_and_no_message_passing_ablation(self):
        frame = events()
        model = DeepCASEGNN(hidden_size=8, embedding_size=4, gnn_layers=0)
        model.fit(frame.iloc[:1], epochs=1)
        self.assertEqual(len(model.context_builder.convs), 0)
        self.assertTrue(np.isfinite(model.score(frame).score).all())
        with self.assertRaises(RuntimeError):
            model.predict(frame)

    def test_metrics_single_class_is_explicit(self):
        metrics = detection_metrics([0, 0], [0.1, 0.9], [0, 1])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["auc_pr"])
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        perfect = detection_metrics([0, 1], [0.1, 0.9], [0, 1])
        self.assertEqual(perfect["auc_pr"], 1)
        self.assertEqual(perfect["average_precision"], 1)


if __name__ == "__main__":
    unittest.main()
