import os
from copy import copy
import ujson

import argparse
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm


def gen_oracle(args, example):
    num_edus = example['num_edus_post_trunc']
    oracle_idxs = list(sorted(example['oracle_idxs']))

    # Add-1
    # Erase-1
    # Swap-1
    candidates = []
    num_to_add = min(args.max_strategy_types, num_edus - len(oracle_idxs))
    num_to_erase = min(args.max_strategy_types, len(oracle_idxs))
    num_to_swap = min(num_to_add, len(oracle_idxs))
    if len(oracle_idxs) == 1:
        num_to_erase = 0  # No empty extracts

    non_oracle_idxs = [int(i) for i in np.arange(num_edus) if i not in oracle_idxs]
    np.random.shuffle(non_oracle_idxs)
    for add_idx in non_oracle_idxs[:num_to_add]:
        new_cand = oracle_idxs + [add_idx]
        candidates.append({
            'strategy': 'add',
            'idxs': new_cand,
        })

    remove_order = np.arange(len(oracle_idxs))
    np.random.shuffle(remove_order)
    for erase_idx in remove_order[:num_to_erase]:
        new_cand = oracle_idxs[:erase_idx] + oracle_idxs[erase_idx + 1:]
        candidates.append({
            'strategy': 'remove',
            'idxs': new_cand,
        })

    np.random.shuffle(non_oracle_idxs)
    for add_idx in non_oracle_idxs[:num_to_swap]:
        remove_loc = int(np.random.randint(len(oracle_idxs)))
        new_cand = copy(oracle_idxs)
        new_cand[remove_loc] = add_idx
        candidates.append({
            'strategy': 'swap',
            'idxs': new_cand
        })

    return {
        'id': example['id'],
        'candidates': candidates
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Random Set of Corrupted for each dataset to be used as calibration guides')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='validation,test,train')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--oracle_col', default='oracle_idxs')
    parser.add_argument('--max_strategy_types', default=8, type=int)

    args = parser.parse_args()

    print(f'Loading {args.dataset}...')
    data_dir = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    dataset = load_from_disk(data_dir)

    out_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    print(f'Creating directory to store pre-computed oracles -> {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(','):
        data_split = dataset[split]
        if args.debug:
            data_split = data_split.select(list(range(16)))
        print(f'Processing {len(data_split)} {split} examples')

        outputs = list(tqdm(map(
            lambda example: gen_oracle(args, example=example),
            data_split), total=len(data_split)))

        out_fn = os.path.join(out_dir, f'{split}_candidates.json')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs_by_id = {arr['id']: arr['candidates'] for arr in outputs}
        with open(out_fn, 'w') as fd:
            ujson.dump(outputs_by_id, fd)
