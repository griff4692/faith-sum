from itertools import combinations
import os
import ujson
import regex as re

import argparse
from datasets import load_dataset, load_metric, load_from_disk
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

    avg_rs = np.array([(a + b) / 2.0 for a, b in zip(r1_hist, r2_hist)])
    sent_priority = np.argsort(-avg_rs)
    n = len(sent_priority)
    top_k_sents = sent_priority[:min(n, args.k)]
    candidates = list(combinations(top_k_sents, 3))
    np.random.shuffle(candidates)
    sent_plans = candidates[:args.num_candidates]
    sent_plans = [tuple(map(int, x)) for x in sent_plans]

    extracts = [' '.join([source_sents[i] for i in idx]) for idx in sent_plans]
    rouges = [
        compute_rouge(rouge, extract, target) for extract in extracts
    ]

    output = {
        'id': example['id'],
        'candidates': []
    }
    for extract_idx, extract, rouge in zip(sent_plans, extracts, rouges):
        row = rouge
        row['mean_f1'] = (rouge['rouge1_f1'] + rouge['rouge1_f1']) / 2.0
        row['extract'] = extract
        row['extract_idx'] = extract_idx
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

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--num_candidates', default=5, type=int)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()
    rouge = load_metric('rouge')

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        data_dir = os.path.join(args.data_dir, args.dataset)
        dataset = load_from_disk(data_dir)
    else:
        dataset = load_dataset(args.dataset)

    out_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    print(f'Creating directory to store pre-computed oracles -> {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(','):
        data_split = dataset[split]
        if args.debug:
            data_split = data_split.select(list(range(128)))
        print(f'Processing {len(data_split)} {split} examples')
        if args.debug:
            outputs = list(tqdm(map(
                lambda example: gen_oracle(args, example=example, rouge=rouge),
                data_split), total=len(data_split)))
        else:
            outputs = list(p_uimap(
                lambda example: gen_oracle(args, example=example, rouge=rouge), data_split
            ))
        out_fn = os.path.join(out_dir, f'{split}_candidates_v2.json')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs_by_id = {arr['id']: arr['candidates'] for arr in outputs}
        with open(out_fn, 'w') as fd:
            ujson.dump(outputs_by_id, fd)
