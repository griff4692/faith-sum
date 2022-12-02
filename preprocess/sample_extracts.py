import os
from copy import copy
import ujson
import regex as re

import argparse
from datasets import load_metric, load_from_disk
from p_tqdm import p_uimap
import numpy as np
from tqdm import tqdm

from sum_constants import summarization_name_mapping


def gen_oracle(args, example, rouge):
    input_col, target_col = summarization_name_mapping[args.dataset]
    inputs = example['source_annotated'].strip()
    target = example[target_col].strip()
    r1_hist = [float(x) for x in example['rouge1_history'].split('|')[0].split(',')]
    r2_hist = [float(x) for x in example['rouge2_history'].split('|')[0].split(',')]

    tps = re.split(r'(<s\d+>)', inputs)
    source_sents = [
        s.strip() for i, s in enumerate(tps) if i > 0 if tps[i - 1].startswith('<s') and tps[i - 1].endswith('>')
    ]

    oracle_idxs = example['oracle_idxs']

    # Add-1
    # Erase-1
    # Swap-1
    candidates = []
    num_to_add = min(args.max_strategy_types, len(source_sents) - len(oracle_idxs))
    num_to_erase = min(args.max_strategy_types, len(oracle_idxs))
    num_to_swap = num_to_add
    if len(oracle_idxs) == 1:
        num_to_erase = 0  # No empty extracts

    avg_rs = np.array([(a + b) / 2.0 for a, b in zip(r1_hist, r2_hist)])
    for idx in oracle_idxs:
        avg_rs[idx] = float('-inf')
    sent_priority = np.argsort(-avg_rs)
    add_priority = sent_priority.copy()
    for add_idx in add_priority[:num_to_add]:
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

    for swap_idx in range(num_to_swap):
        add_idx = add_priority[swap_idx]
        remove_loc = int(np.random.randint(len(oracle_idxs)))
        new_cand = copy(oracle_idxs)
        new_cand[remove_loc] = add_idx
        candidates.append({
            'strategy': 'swap',
            'idxs': new_cand
        })

    extracts = [' '.join([source_sents[i] for i in obj['idxs']]) for obj in candidates]
    rouges = [
        compute_rouge(rouge, extract, target) for extract in extracts
    ]

    output = {
        'id': example['id'],
        'candidates': []
    }
    for candidate, extract, rouge in zip(candidates, extracts, rouges):
        row = rouge
        row['mean_f1'] = float((rouge['rouge1_f1'] + rouge['rouge1_f1']) / 2.0)
        row['extract'] = extract
        row['extract_idx'] = [int(x) for x in sorted(candidate['idxs'])]
        row['strategy'] = candidate['strategy']
        output['candidates'].append(row)

    output['candidates'] = list(sorted(output['candidates'], key=lambda x: -x['mean_f1']))
    return output


def compute_rouge(rouge, summary, reference):
    rouge_types = ['rouge1', 'rouge2']
    rouge_output = rouge.compute(predictions=[summary], references=[reference], rouge_types=['rouge1', 'rouge2'])
    stats = {}
    for rouge_type in rouge_types:
        stats[f'{rouge_type}_precision'] = float(rouge_output[rouge_type].mid.precision)
        stats[f'{rouge_type}_recall'] = float(rouge_output[rouge_type].mid.recall)
        stats[f'{rouge_type}_f1'] = float(rouge_output[rouge_type].mid.fmeasure)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--splits', default='validation,test,train')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('--max_strategy_types', default=10, type=int)

    args = parser.parse_args()
    rouge = load_metric('rouge')

    print(f'Loading {args.dataset}...')
    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)

    out_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    print(f'Creating directory to store pre-computed oracles -> {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(','):
        data_split = dataset[split]
        if args.debug:
            data_split = data_split.select(list(range(16)))
        print(f'Processing {len(data_split)} {split} examples')
        if args.debug:
            outputs = list(tqdm(map(
                lambda example: gen_oracle(args, example=example, rouge=rouge),
                data_split), total=len(data_split)))
        else:
            outputs = list(p_uimap(
                lambda example: gen_oracle(args, example=example, rouge=rouge), data_split, num_cpus=args.cpu_frac
            ))
        out_fn = os.path.join(out_dir, f'{split}_candidates.json')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs_by_id = {arr['id']: arr['candidates'] for arr in outputs}
        with open(out_fn, 'w') as fd:
            ujson.dump(outputs_by_id, fd)
