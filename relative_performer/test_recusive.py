#!/usr/bin/env python3
"""Test runs in a nested folder structure and combine results into a csv."""
import argparse
from pathlib import Path

import pandas as pd

from relative_performer.test import test_run, NoCheckpointFoundException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=Path)
    parser.add_argument('--output', type=Path)

    args = parser.parse_args()
    run_dirs = Path(args.run_dirs)

    data = []
    for hparam_file in run_dirs.rglob('hparams.yaml'):
        try:
            run_dir = hparam_file.parent
            results = test_run(run_dir)
            data.append(pd.DataFrame(results, index=[run_dir]))
        except NoCheckpointFoundException:
            pass

    output = pd.concat(data, axis=0)
    output.index.name = 'run_dir'
    output.to_csv(args.output)
