"""Stream a complete LANL interval to disk and report exact label coverage."""
import argparse
import gzip
import json
from pathlib import Path


def prepare(auth_path, redteam_path, output, start, end, split_time):
    if not 0 <= start < split_time < end:
        raise ValueError('Require 0 <= start < split-time < end.')
    opener = gzip.open if str(redteam_path).endswith('.gz') else open
    with opener(redteam_path, 'rt') as stream:
        keys = {tuple(line.strip().split(',')) for line in stream if line.strip()}
    report = {
        'auth': str(Path(auth_path).resolve()),
        'redteam': str(Path(redteam_path).resolve()),
        'interval': {'start_inclusive': start, 'end_exclusive': end,
                     'split_time': split_time},
        'scanned_rows': 0,
    }
    matched = {'train': set(), 'test': set()}
    for name, lower, upper in [('train', start, split_time), ('test', split_time, end)]:
        report[name] = {'rows': 0, 'positive_rows': 0,
                        'redteam_keys_in_interval': sum(lower <= int(k[0]) < upper for k in keys)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(auth_path).endswith('.gz') else open
    # Exclusive creation protects both raw inputs and previous extracts.
    with opener(auth_path, 'rb') as source, output.open('xb') as target:
        for line in source:
            report['scanned_rows'] += 1
            timestamp = int(line.split(b',', 1)[0])
            if start <= timestamp < end:
                fields = line.decode().strip().split(',')
                if len(fields) != 9:
                    raise ValueError('Malformed auth row {}'.format(report['scanned_rows']))
                key = (fields[0], fields[1], fields[3], fields[4])
                name = 'train' if timestamp < split_time else 'test'
                report[name]['rows'] += 1
                if key in keys:
                    report[name]['positive_rows'] += 1
                    matched[name].add(key)
                target.write(line)
            if report['scanned_rows'] % 10_000_000 == 0:
                print('Scanned {:,} rows; retained {:,}'.format(
                    report['scanned_rows'], report['train']['rows'] + report['test']['rows']), flush=True)
    # Scan to EOF: the source is not assumed globally sorted.
    for name in matched:
        report[name]['unique_matched_redteam_keys'] = len(matched[name])
        report[name]['unmatched_redteam_keys'] = report[name]['redteam_keys_in_interval'] - len(matched[name])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auth', type=Path, required=True)
    parser.add_argument('--redteam', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--split-time', type=int, required=True)
    args = parser.parse_args()
    report = prepare(args.auth, args.redteam, args.output, args.start, args.end, args.split_time)
    report_path = args.output.with_suffix(args.output.suffix + '.report.json')
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print('Report:', report_path)


if __name__ == '__main__':
    main()
