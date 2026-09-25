"""User-launched baseline sweep: HDFS first, then LANL, in isolated processes.

From the repository root:
    .venv/bin/python example/run_experiments.py
    .venv/bin/python example/run_experiments.py --datasets hdfs
    .venv/bin/python example/run_experiments.py --datasets lanl

Defaults: independent 25/50/100-epoch runs, seed 42, CUDA required. HDFS uses
1000 sequences (next-event prediction); --hdfs-nrows 0 reads all sequences.
LANL reads the complete existing auth_144000_154000.txt extract and splits at
147600. This is an interpreter diagnostic, NOT the full paper protocol or a
GNN experiment. Its benign-only cluster scores cannot flag positive attacks.
Use --lanl-auth to select another prepared interval; the example loads it into
RAM. This runner does not extract intervals or silently change the split.

Each invocation creates a new results directory. Failure or interruption stops
before the next experiment; consult status.json and the affected run.log.
--plan prints the commands and performs no training or output-directory writes.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'example'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def script_arguments(args, dataset, epochs):
    command = [str(EXAMPLE / f'example_{dataset}.py'), '--epochs', str(epochs)]
    if dataset == 'hdfs':
        command += ['--no-graphs']
        if args.hdfs_nrows:
            command += ['--nrows', str(args.hdfs_nrows)]
    else:
        command += ['--auth', str(args.lanl_auth), '--redteam', str(args.redteam),
                    '--nrows', str(args.lanl_nrows), '--split-time', str(args.split_time),
                    '--auth-scope', 'remote-logon', '--batch-size', str(args.batch_size),
                    '--query-batch-size', str(args.query_batch_size), '--verbose']
    return command


def worker(args):
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from plotting import ResultsPlotter

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. No CPU fallback is performed.')
    dataset, epochs = args.worker
    epochs = int(epochs)
    output = args.output / dataset / str(epochs)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    command = script_arguments(args, dataset, epochs)
    save_json(output / 'config.json', {
        'dataset': dataset, 'epochs': epochs, 'seed': args.seed, 'argv': command,
        'task': 'next-event prediction' if dataset == 'hdfs' else 'interpreter diagnostic',
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'python': sys.version,
    })
    print('Starting:', ' '.join(command), flush=True)
    start = time.monotonic()
    sys.argv = command
    state = runpy.run_path(command[0], run_name='__main__')
    builder = state['context_builder'] if dataset == 'hdfs' else state['deepcase'].context_builder
    losses = [float(value) for value in builder.loss_history]
    if len(losses) != epochs:
        raise RuntimeError(f'Training stopped early: {len(losses)} of {epochs} epochs; not marking complete.')
    actual, predicted = state['y_test'], state['y_pred']
    arrays = {'y_test': actual, 'y_pred': predicted}
    metrics = {
        'classification_report': classification_report(actual, predicted, output_dict=True, zero_division=0),
        'elapsed_seconds': time.monotonic() - start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'test_rows': len(actual),
    }
    if dataset == 'lanl':
        frame, split = state['data'], state['split']
        test = frame.iloc[split:]
        raw = state['prediction']
        arrays.update(raw_scores=raw, event_id=test.event_id.to_numpy(), timestamp=test.timestamp.to_numpy())
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        metrics.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                       unscored_events=int((raw < 0).sum()),
                       raw_score_counts={str(k): int(v) for k, v in zip(*np.unique(raw, return_counts=True))},
                       caveat='Benign-only interpreter diagnostic; negative/unscored outputs become negative classes under prediction > 0. No unsupervised AP claimed.')
        test[['event_id', 'timestamp', 'source_computer', 'destination_computer', 'label']].to_csv(
            output / 'test_events.csv.gz', index=False)
        partitions = {}
        for name, part in [('train', frame.iloc[:split]), ('test', test)]:
            partitions[name] = {'rows': len(part), 'positive_rows': int(part.label.sum()),
                                'first_timestamp': int(part.timestamp.min()), 'last_timestamp': int(part.timestamp.max())}
        save_json(output / 'split_manifest.json', partitions)
    else:
        arrays['event_index'] = np.arange(len(state['events_train']), len(state['events']))
        save_json(output / 'split_manifest.json', {
            'train_rows': len(state['events_train']), 'test_rows': len(actual),
            'policy': 'Existing HDFS example: first 20% of generated events train, remaining 80% test. Not anomaly-label evaluation.'})
    np.savez_compressed(output / 'predictions.npz', **arrays)
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'loss_history.json', losses)
    save_json(output / 'event_mapping.json', {str(k): int(v) for k, v in state['mapping'].items()})
    torch.save(builder.state_dict(), output / 'context_builder.pt')
    ResultsPlotter.plot_prediction_results(actual, predicted,
        output / f'{dataset}_prediction_results.png', show=False,
        title=f'{dataset.upper()} — {epochs} epochs' + (' (interpreter diagnostic)' if dataset == 'lanl' else ' (next event)'))
    ResultsPlotter.plot_training_loss_history(losses, output / f'{dataset}_training_loss_summary.png', show=False)
    print(f'Completed {dataset}, {epochs} epochs: {output}', flush=True)


def summarize(output, status):
    lines = ['# Baseline sweep', '',
             'HDFS measures next-event prediction. LANL is an interpreter diagnostic;',
             'these tasks and their accuracy values are not directly comparable.', '',
             '| Run | Status | Accuracy | Final recorded loss | Seconds |',
             '|---|---|---:|---:|---:|']
    for key, value in status.items():
        if value == 'complete':
            folder = output / key
            metrics = json.loads((folder / 'metrics.json').read_text())
            losses = json.loads((folder / 'loss_history.json').read_text())
            lines.append(f'| [{key}]({key}/metrics.json) | complete | {metrics["classification_report"]["accuracy"]:.6f} | {losses[-1]:.6f} | {metrics["elapsed_seconds"]:.1f} |')
        else:
            lines.append(f'| {key} | {value} | — | — | — |')
    lines += ['', 'For LANL, inspect TP/FN and unscored counts in metrics.json; high accuracy',
              'alone does not establish attack detection. Full vocabulary/validation/scoring',
              'protocol work remains separate from this baseline rerun.', '']
    (output / 'summary.md').write_text('\n'.join(lines))


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', choices=['hdfs', 'lanl'], nargs='+', default=['hdfs', 'lanl'])
    parser.add_argument('--epochs', type=int, nargs='+', default=[25, 50, 100])
    parser.add_argument('--output', type=Path, help='New directory only; default: example/results/<timestamp>_hdfs_lanl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hdfs-nrows', type=int, default=1000, help='HDFS sequences; 0 = entire file (higher memory use).')
    parser.add_argument('--lanl-auth', type=Path, default=EXAMPLE / 'data/lanl/auth_144000_154000.txt')
    parser.add_argument('--redteam', type=Path, default=EXAMPLE / 'data/lanl/redteam.txt')
    parser.add_argument('--lanl-nrows', type=int, default=0, help='LANL physical rows; 0 = entire selected file.')
    parser.add_argument('--split-time', type=int, default=147600)
    parser.add_argument('--batch-size', type=int, default=128, help='LANL training batch size.')
    parser.add_argument('--query-batch-size', type=int, default=128)
    parser.add_argument('--plan', action='store_true', help='Print plan without running experiments or writing files.')
    parser.add_argument('--worker', nargs=2, metavar=('DATASET', 'EPOCHS'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(e < 1 for e in args.epochs) or len(set(args.epochs)) != len(args.epochs):
        parser.error('Epoch counts must be positive and unique.')
    if min(args.hdfs_nrows, args.lanl_nrows) < 0 or min(args.batch_size, args.query_batch_size) < 1:
        parser.error('Row limits must be nonnegative; batch sizes must be positive.')
    args.lanl_auth, args.redteam = args.lanl_auth.resolve(), args.redteam.resolve()
    args.output = (args.output or EXAMPLE / 'results' / (datetime.now().strftime('%Y-%m-%d_%H%M%S_%f') + '_hdfs_lanl')).resolve()
    os.chdir(ROOT)
    if args.worker:
        worker(args)
        return
    jobs = [(dataset, epochs) for dataset in ['hdfs', 'lanl'] if dataset in args.datasets for epochs in args.epochs]
    print('Results:', args.output, flush=True)
    for dataset, epochs in jobs:
        print(' '.join(script_arguments(args, dataset, epochs)), flush=True)
    if args.plan:
        return
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA is required; no experiments started.')
    inputs = []
    if 'hdfs' in args.datasets:
        inputs.append(EXAMPLE / 'data/hdfs/hdfs_test_normal')
    if 'lanl' in args.datasets:
        inputs.extend([args.lanl_auth, args.redteam])
    for path in inputs:
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    status = {f'{d}/{e}': 'pending' for d, e in jobs}
    save_json(args.output / 'status.json', status)
    summarize(args.output, status)
    save_json(args.output / 'manifest.json', {'inputs': [fingerprint(p) for p in inputs],
              'invocation': sys.argv, 'seed': args.seed,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
    (args.output / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], text=True))
    for source in [Path(__file__).resolve(), EXAMPLE / 'example_hdfs.py', EXAMPLE / 'example_lanl.py']:
        (args.output / source.name).write_text(source.read_text())
    for dataset, epochs in jobs:
        key = f'{dataset}/{epochs}'
        folder = args.output / key
        folder.mkdir(parents=True)
        status[key] = 'running'
        save_json(args.output / 'status.json', status)
        summarize(args.output, status)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', dataset, str(epochs),
                   '--output', str(args.output), '--seed', str(args.seed),
                   '--hdfs-nrows', str(args.hdfs_nrows), '--lanl-auth', str(args.lanl_auth),
                   '--redteam', str(args.redteam), '--lanl-nrows', str(args.lanl_nrows),
                   '--split-time', str(args.split_time), '--batch-size', str(args.batch_size),
                   '--query-batch-size', str(args.query_batch_size)]
        print(f'\nRunning {key}; log: {folder / "run.log"}', flush=True)
        process = None
        try:
            with (folder / 'run.log').open('x') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f'{key} exited with code {code}; inspect {folder / "run.log"}')
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            status[key] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            save_json(args.output / 'status.json', status)
            summarize(args.output, status)
            raise
        status[key] = 'complete'
        save_json(args.output / 'status.json', status)
        summarize(args.output, status)
    print(f'All requested runs completed. Summary: {args.output / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
