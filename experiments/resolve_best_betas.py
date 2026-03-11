#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiments import load_config, merge_dicts, resolve_dynamic_spec_fields


def main(config_path: Path, output_path: Path) -> None:
    config = load_config(config_path)
    base_config = config.get('base_config', {})
    resolved_experiments = []
    for experiment in config.get('experiments', []):
        spec = merge_dicts(base_config, experiment)
        resolved = resolve_dynamic_spec_fields(spec, workspace_root=Path.cwd())
        resolved_experiments.append(resolved)
    resolved_config = dict(config)
    resolved_config['experiments'] = resolved_experiments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.out)
