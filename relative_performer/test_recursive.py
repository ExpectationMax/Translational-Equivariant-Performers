#!/usr/bin/env python3
"""Test runs in a nested folder structure and combine results into a csv."""
import argparse
from pathlib import Path
from glob import glob

import pandas as pd

from relative_performer.test import test_run, NoCheckpointFoundException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=Path)
    parser.add_argument('--output', type=Path)

    args = parser.parse_args()
    run_dirs = Path(args.run_dirs)

    data = []
    # Need to use glob instead of Path.rglob as it does not follow symlinks
    for hparam_file in glob(str(run_dirs.joinpath('**','hparams.yaml')), recursive=True):
        try:
            run_dir = Path(hparam_file).parent
            results = test_run(run_dir)
            data.append(pd.DataFrame(results, index=[str(run_dir)]))
        except NoCheckpointFoundException:
            pass

    output = pd.concat(data, axis=0)
    output.index.name = 'run_dir'
    output.to_csv(args.output)
