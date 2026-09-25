"""Run isolated LANL interpreter diagnostics; never overwrite an existing run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ['MPLBACKEND'] = 'Agg'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def worker(args, epochs, output):
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from plotting import ResultsPlotter
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment set.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    command = [str(ROOT / 'example/example_lanl.py'), '--auth', str(args.auth),
               '--redteam', str(args.redteam), '--nrows', '0', '--split-time', '147600',
               '--epochs', str(epochs), '--batch-size', '128', '--query-batch-size', '128']
    write_json(output / 'config.json', {'argv': command, 'seed': args.seed,
               'interpretation': 'interpreter diagnostic, not unsupervised detection',
               'training_interval': [144000, 147600], 'test_interval': [147600, 154000]})
    print('Arguments:', command, flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(f'Run active: {time.monotonic() - start:.0f}s elapsed', flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    sys.argv = command
    try:
        state = runpy.run_path(command[0], run_name='__main__')
    finally:
        stop.set()
    data, split = state['data'], state['split']
    actual, predicted, raw = state['y_test'], state['y_pred'], state['prediction']
    test = data.iloc[split:]
    np.savez_compressed(output / 'predictions.npz', y_test=actual, y_pred=predicted,
                        raw_scores=raw, event_id=test.event_id.to_numpy(),
                        timestamp=test.timestamp.to_numpy())
    test[['event_id', 'timestamp', 'source_computer', 'destination_computer', 'label']].to_csv(
        output / 'test_events.csv.gz', index=False)
    counts = {}
    for name, frame in [('train', data.iloc[:split]), ('test', test)]:
        counts[name] = {'rows': len(frame), 'positive_rows': int(frame.label.sum()),
                        'first_timestamp': int(frame.timestamp.min()),
                        'last_timestamp': int(frame.timestamp.max())}
    write_json(output / 'split_manifest.json', counts)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    metrics = {'classification_report': classification_report(actual, predicted,
                labels=[0, 1], output_dict=True, zero_division=0),
               'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
               'fpr': float(fp / (fp + tn)), 'unscored_events': int((raw < 0).sum()),
               'raw_score_counts': {str(k): int(v) for k, v in zip(*np.unique(raw, return_counts=True))},
               'elapsed_seconds': time.monotonic() - start,
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
               'caveat': 'prediction > 0 diagnostic treats unscored events as negative; no anomaly AP claimed'}
    write_json(output / 'metrics.json', metrics)
    builder = state['deepcase'].context_builder
    write_json(output / 'loss_history.json', [float(v) for v in builder.loss_history])
    torch.save(builder.state_dict(), output / 'context_builder.pt')
    write_json(output / 'event_mapping.json', {str(k): int(v) for k, v in state['mapping'].items()})
    ResultsPlotter.plot_prediction_results(actual, predicted,
        output / 'lanl_prediction_results.png', show=False,
        title=f'LANL interpreter diagnostic — {epochs} epochs')
    ResultsPlotter.plot_training_loss_history(builder.loss_history,
        output / 'lanl_training_loss_summary.png', show=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auth', type=Path, default=ROOT / 'example/data/lanl/auth_144000_154000.txt')
    parser.add_argument('--redteam', type=Path, default=ROOT / 'example/data/lanl/redteam.txt')
    parser.add_argument('--epochs', type=int, nargs='+', default=[25, 50, 100])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    args.output, args.auth, args.redteam = (p.resolve() for p in (args.output, args.auth, args.redteam))
    if args.worker is not None:
        worker(args, args.worker, args.output)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse reuse before changing any existing artifacts.
    if (args.output / 'manifest.json').exists() or any((args.output / str(e)).exists() for e in args.epochs):
        raise FileExistsError('Choose a new output folder for this experiment set.')
    write_json(args.output / 'manifest.json', {
        'auth': fingerprint(args.auth), 'redteam': fingerprint(args.redteam),
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'epochs': args.epochs, 'seed': args.seed,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'bounded remote LogOn pilot, not full first-41-hours training',
    })
    (args.output / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
    for source in [Path(__file__), ROOT / 'example/example_lanl.py']:
        (args.output / source.name).write_text(source.read_text())
    statuses = {}
    for epochs in args.epochs:
        output = args.output / str(epochs)
        output.mkdir()
        statuses[str(epochs)] = 'running'
        write_json(args.output / 'status.json', statuses)
        print(f'Starting {epochs} epochs: {output}', flush=True)
        with (output / 'run.log').open('x') as log:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(output), '--auth', str(args.auth), '--redteam', str(args.redteam),
                '--seed', str(args.seed), '--worker', str(epochs)], stdout=log, stderr=subprocess.STDOUT)
        statuses[str(epochs)] = 'complete' if result.returncode == 0 else 'failed'
        write_json(args.output / 'status.json', statuses)
        if result.returncode:
            raise RuntimeError(f'{epochs}-epoch run failed; see {output / "run.log"}')
    print('All runs complete.', flush=True)


if __name__ == '__main__':
    main()
