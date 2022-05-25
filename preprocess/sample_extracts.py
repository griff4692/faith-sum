from itertools import combinations
import os
import ujson

import argparse
from p_tqdm import p_uimap
import numpy as np
import spacy
from tqdm import tqdm

from preprocess.convert_abstractive_to_extractive import gain_selection
from sum_constants import summarization_name_mapping
from datasets import load_dataset
from preprocess.extract_oracles import convert_to_sents
from datasets import load_metric


def gen_oracle(args, example, nlp, rouge, k=10):
    input_col, target_col = summarization_name_mapping[args.dataset]
    inputs = example[input_col].strip()
    target = example[target_col].strip()
    source_sents = convert_to_sents(inputs, nlp)
    source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
    target_sents = convert_to_sents(target, nlp)
    target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
    # Sort oracle order or not
    oracle_idxs, oracle_rouge, r1_hist, r2_hist, best_hist = gain_selection(
        source_sents_tok, target_sents_tok, 0, lower=True, sort=False)

    # priority = [(r1 + r2) / 2.0 for r1, r2 in zip(r1_hist.split('|'), r2_hist)]
    r1s = [float(x) for x in r1_hist.split('|')[0].split(',')]
    r2s = [float(x) for x in r2_hist.split('|')[0].split(',')]
    avg_rs = np.array([(a + b) / 2.0 for a, b in zip(r1s, r2s)])
    sent_priority = np.argsort(-avg_rs)
    n = len(sent_priority)
    top_k_sents = sent_priority[:min(n, k)]
    candidates = list(combinations(top_k_sents, 3))
    np.random.shuffle(candidates)
    sent_plans = candidates[:16]

    extracts = [' '.join([str(source_sents[i]) for i in idx]) for idx in sent_plans]
    rouges = [
        compute_rouge(rouge, extract, target) for extract in extracts
    ]

    output = {
        'id': example['id'],
        'candidates': []
    }
    for extract, rouge in zip(extracts, rouges):
        row = rouge
        row['mean_f1'] = (rouge['rouge1_f1'] + rouge['rouge1_f1']) / 2.0
        row['extract_idx'] = extract
        output['candidates'].append(row)

    return output


def compute_rouge(rouge, summary, reference):
    rouge_types = ['rouge1', 'rouge2']
    rouge_output = rouge.compute(predictions=[summary], references=[reference], rouge_types=['rouge1', 'rouge2'])
    stats = {}
    for rouge_type in rouge_types:
        stats[f'{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
        stats[f'{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
        stats[f'{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()
    args.debug = True

    rouge = load_metric('rouge')

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

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
                lambda example: gen_oracle(args, example=example, nlp=nlp, rouge=rouge),
                data_split), total=len(data_split)))
        else:
            outputs = list(p_uimap(
                lambda example: gen_oracle(args, example=example, nlp=nlp, rouge=rouge), data_split
            ))
        out_fn = os.path.join(out_dir, f'{split}_candidates.json')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs_by_id = {arr['id']: arr['candidates'] for arr in outputs}
        with open(out_fn, 'w') as fd:
            ujson.dump(outputs_by_id, fd)
