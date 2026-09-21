"""Compare two builds in alternate process order. Keep all timing samples."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder', type=Path)
parser.add_argument('--kind', choices=['ggml', 'tract'], required=True)
parser.add_argument('--pairs', type=int, default=4)
parser.add_argument('--threads', type=int, nargs='+', default=[1])
parser.add_argument('--output', default='comparison.json')
parser.add_argument('--control', action='store_true', help='Use the base binary in both slots.')
parser.add_argument('--base-name', default='bench_base')
parser.add_argument('--candidate-name', default='bench_candidate')
args = parser.parse_args()
folder = args.folder.resolve()
result = dict(kind=args.kind, pairs=args.pairs, control=args.control, binaries={}, cases={}, runs=[])
paths = {'base': folder / args.base_name,
         'candidate': folder / (args.base_name if args.control else args.candidate_name)}
for variant in ['base', 'candidate']:
    result['binaries'][variant] = hashlib.sha256(paths[variant].read_bytes()).hexdigest()
start = time.monotonic()
for threads in args.threads:
    for pair in range(args.pairs):
        for variant in (['base', 'candidate'] if pair % 2 == 0 else ['candidate', 'base']):
            command = ['taskset', '-c', ','.join(map(str, range(threads))), str(paths[variant])]
            if args.kind == 'ggml':
                command += ['bench', str(threads)]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            groups = {}
            for row in csv.DictReader(io.StringIO(run.stdout)):
                fields = {k: v for k, v in row.items() if k not in ('us', 'sample', 'hash')}
                fields['threads'] = threads
                key = json.dumps(fields, sort_keys=True)
                entry = result['cases'].setdefault(key, dict(fields=fields, base=[], candidate=[], hash=row['hash']))
                if entry['hash'] != row['hash']:
                    raise RuntimeError(f'Output hashes differ: {key}')
                groups.setdefault(key, []).append(float(row['us']))
            for key, samples in groups.items():
                result['cases'][key][variant].append(samples)
            result['runs'].append(dict(variant=variant, threads=threads, pair=pair, stderr=run.stderr.strip()))
            print(args.kind, 'threads', threads, 'pair', pair, variant, flush=True)
for entry in result['cases'].values():
    for variant in ['base', 'candidate']:
        medians = [statistics.median(samples) for samples in entry[variant]]
        entry[variant + '_us'] = statistics.median(medians)
        entry[variant + '_range_us'] = [min(medians), max(medians)]
    entry['reduction_percent'] = 100 * (1 - entry['candidate_us'] / entry['base_us'])
    entry['paired_reduction_percent'] = [100 * (1 - statistics.median(b) / statistics.median(a))
                                         for a, b in zip(entry['base'], entry['candidate'])]
result['elapsed_s'] = time.monotonic() - start
(folder / args.output).write_text(json.dumps(result, indent=2) + '\n')
