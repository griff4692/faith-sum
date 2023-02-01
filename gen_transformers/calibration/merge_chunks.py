from glob import glob
import os
import ujson
import numpy as np
import regex as re

import argparse


def merge(fns):
    obj = []
    for fn in fns:
        with open(fn, 'r') as fd:
            obj += ujson.load(fd)
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate From Extract')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--dataset', default='cnn_dailymail')

    args = parser.parse_args()

    oracle_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    for split in args.splits.split(','):
        print(split)
        pattern = os.path.join(oracle_dir, f'{split}_candidates_targets_chunk_*.json')
        fns = list(glob(pattern))
        chunk_idxs = [int(re.search(r'_candidates_targets_chunk_(\d+)', x).group(1)) for x in fns]
        order = list(np.argsort(chunk_idxs))
        fns = [fns[i] for i in order]
        fn_str = '\n'.join(fns)
        print(f'Merging Following Files:\n{fn_str}')

        full_obj = merge(fns)
        out_fn = os.path.join(oracle_dir, f'{split}_candidates_targets.json')
        print(f'Saving merged of length {len(full_obj)} to {out_fn}')
        with open(out_fn, 'w') as fd:
            ujson.dump(full_obj, fd)
