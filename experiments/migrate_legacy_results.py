#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.result_utils import RESULT_FIELDS

LEGACY_FIELDS = ["exp_name", "dataset", "alpha", "lambda_scale", "epoch", "nfe", "fid", "status"]


def load_legacy_rows(path: Path):
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEGACY_FIELDS:
            raise ValueError(
                f'Unsupported legacy schema in {path}. Expected {LEGACY_FIELDS}, got {reader.fieldnames}.'
            )
        return list(reader)


def migrate_row(row, artifact_group: str, index: int):
    alpha = float(row['alpha'])
    is_baseline = abs(alpha - 0.5) < 1e-8 and row.get('lambda_scale', '') == 'auto'
    return {
        'run_id': f'legacy:{artifact_group}:{index}',
        'exp_name': row['exp_name'],
        'dataset': row['dataset'],
        'seed': 0,
        'stage': 'eval',
        'checkpoint_epoch': int(row['epoch']),
        'path_family': 'linear',
        'clock_family': 'uniform' if is_baseline else 'legacy_alpha',
        'clock_param_name': 'none' if is_baseline else 'alpha',
        'clock_param_value': '' if is_baseline else alpha,
        'solver': 'unknown',
        'nfe': int(row['nfe']),
        'step_count': 0,
        'real_samples': 0,
        'synthetic_samples': 0,
        'metric': 'fid',
        'value': float(row['fid']),
        'status': row['status'],
        'artifact_group': artifact_group,
    }


def main(src: Path, out: Path) -> None:
    rows = load_legacy_rows(src)
    artifact_group = src.parent.name
    migrated = [migrate_row(row, artifact_group, index) for index, row in enumerate(rows)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(migrated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.src, args.out)
