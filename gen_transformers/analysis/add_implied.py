import os

import argparse
from datasets import load_from_disk
import pandas as pd
import numpy as np
import spacy
from p_tqdm import p_uimap

from eval.rouge_metric import RougeMetric
from gen_transformers.data_utils import infer_dataset
from gen_transformers.gen_from_extract import compute_implied, get_extract_idxs_from_str, compute_rouge


os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


def render(df):
    cols = [
        'official_plan_p',
        'official_plan_r',
        'official_plan_f1',
        'avg_implied_official_rouge1_f1',
        'avg_implied_official_rouge2_f1',
        'avg_implied_official_rougeL_f1'
    ]

    for col in cols:
        if col in df.columns:
            print(col, round(df[col].dropna().mean(), 3))



def _add_implied(record):
    source_annotated = all_source_annotated[record['dataset_idx']]
    # Get source tokens
    reference = record['reference']
    pred_col = 'from_extract_abstract' if 'from_extract_abstract' in record else 'abstract'
    abstracts = record[pred_col].split('<cand>')

    implied_extracts = compute_implied(args, nlp, abstracts, source_annotated)
    implied_idxs = [x['idxs'] for x in implied_extracts]
    implied_summaries = [x['summary'] for x in implied_extracts]
    rouge_comb = {'implied_official_rouge1_f1': [], 'implied_official_rouge2_f1': [], 'implied_official_rougeL_f1': []}
    for summary in implied_summaries:
        rouge = compute_rouge([summary], [reference], rouge_metric, prefix='implied_official_')
        for k in rouge_comb:
            rouge_comb[k].append(rouge[k])
    rcols = list(rouge_comb.keys())
    for k in rcols:
        v = rouge_comb[k]
        rouge_comb[k] = ','.join(map(str, v))
        rouge_comb['avg_' + k] = np.mean(v)
    record.update(rouge_comb)
    implied_str = '<cand>'.join([
        ','.join([str(x) for x in z]) for z in implied_idxs
    ])
    record['implied_extract_idx'] = implied_str

    if pred_col == 'from_extract_abstract':
        extract_idxs = list(map(get_extract_idxs_from_str, record['extract_idx'].split('<cand>')))
        ps, rs, f1s = [], [], []
        for extract_idx, implied_idx in zip(extract_idxs, implied_idxs):
            agreement = set(extract_idx).intersection(implied_idx)
            n = len(agreement)
            r = n / len(extract_idx)
            p = n / len(implied_idx)
            f1 = 0 if min(r, p) == 0 else (2 * p * r) / (p + r)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        mean_p = np.mean(ps)
        mean_r = np.mean(rs)
        mean_f1 = np.mean(f1s)
        record['official_plan_p'] = mean_p
        record['official_plan_r'] = mean_r
        record['official_plan_f1'] = mean_f1
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add Implied Extract')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--experiment', default='cnn_e_v1')
    parser.add_argument('--fn', default='test_from_beam_16_extract_cnn_ea_rand_v2.csv')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    if not args.fn.endswith('.csv'):
        args.fn += '.csv'

    infer_dataset(args, 'experiment')
    if 'test' in args.fn:
        args.split = 'test'
    elif 'validation' in args.fn:
        args.split = 'validation'
    elif 'train' in args.fn:
        args.split = 'train'
    else:
        raise Exception(f'This is an Exception. Include split name in filename -> {args.fn}!!!')

    in_fn = os.path.join(args.data_dir, 'results', args.experiment, args.fn)
    print(f'Reading in extracts from {in_fn}')
    outputs = pd.read_csv(in_fn)

    if 'avg_implied_official_rouge1_f1' in list(outputs.columns) and not args.overwrite:
        render(outputs)
        print('Already run. Run with -overwrite to re-run.')
        exit(0)

    data_dir = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    print(f'Loading dataset from {data_dir}')
    dataset = load_from_disk(data_dir)[args.split]
    dataset_idx2id = dataset['id']
    all_source_annotated = dataset['source_edu_annotated']

    records = outputs.to_dict('records')

    rouge_metric = RougeMetric()
    nlp = spacy.load('en_core_web_sm')

    updated_records = list(p_uimap(_add_implied, records))
    updated_df = pd.DataFrame(updated_records).sort_values(by='dataset_idx').reset_index(drop=True)
    print(f'Saving {len(updated_df)} back to {in_fn}')
    updated_df.to_csv(in_fn, index=False)

    render(updated_df)
