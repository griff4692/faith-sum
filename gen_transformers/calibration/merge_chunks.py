from glob import glob
import os
import ujson
import numpy as np
import regex as re

import argparse


def merge(fns):
    obj = {}
    n = 0
    for fn in fns:
        with open(fn, 'r') as fd:
            chunk = ujson.load(fd)
            assert type(chunk) == dict
            n += len(chunk)
            obj.update(chunk)
    assert len(obj) == n  # Otherwise ther are duplicate dataset_ids in the chunks
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate From Extract')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--split', default='train')
    parser.add_argument('--dataset', default='cnn_dailymail')

    args = parser.parse_args()

    oracle_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    print(args.split)
    out_fn = os.path.join(oracle_dir, f'{args.split}_candidates_targets.json')
    if os.path.exists(out_fn):
        print(f'rm {out_fn}')
        print('... and re-run')
        exit(0)
    pattern = os.path.join(oracle_dir, f'{args.split}_candidates_targets_chunk_*.json')
    fns = list(glob(pattern))
    chunk_idxs = [int(re.search(r'_candidates_targets_chunk_(\d+)', x).group(1)) for x in fns]
    order = list(np.argsort(chunk_idxs))
    fns = [fns[i] for i in order]
    fn_str = '\n'.join(fns)
    print(f'Merging Following Files:\n{fn_str}')

    full_obj = merge(fns)
    print(f'Saving merged of length {len(full_obj)} to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(full_obj, fd)
