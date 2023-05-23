import os
from glob import glob

import argparse
import numpy as np
import pandas as pd
from evaluate import load
from nltk import word_tokenize, sent_tokenize
from datasets import load_from_disk
from tqdm import tqdm

from eval.diversity import diversity_score


def get_dataset_idx(x):
    return x.split('/')[-1].split('_')[0]


def compute_stats(prediction, reference, rouge):
    rouge_obj = rouge.compute(predictions=[prediction], references=[reference], use_aggregator=False)
    num_toks = len(word_tokenize(prediction))
    num_sents = len(sent_tokenize(prediction))
    return {
        'num_toks': num_toks,
        'num_sents': num_sents,
        'rouge1': rouge_obj['rouge1'][0],
        'rouge2': rouge_obj['rouge2'][0]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-3.5 / GPT-4.5')

    # Configuration Parameters
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--extract_experiment', default='cnn_e_final')
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--max_examples', default=16, type=int)

    args = parser.parse_args()

    results_dir = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, args.model)

    fns = list(glob(os.path.join(results_dir, '*.txt')))

    dataset_idxs = list(sorted(list(set([get_dataset_idx(x) for x in fns]))))

    rouge = load('rouge', keep_in_memory=True)
    dataset = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')
    test = dataset['test']
    idx2target = dict(zip(test['dataset_idx'], test['target']))

    agg_stats = []

    for dataset_idx in tqdm(dataset_idxs):
        idx_fns = list(sorted([fn for fn in fns if get_dataset_idx(fn) == dataset_idx]))
        cands = [open(fn).read().strip() for fn in idx_fns]

        reference = idx2target[int(dataset_idx)]

        stats = [
            compute_stats(cand, reference, rouge)
            for cand in cands
        ]

        stat_df = pd.DataFrame(stats)

        agg_stats.append({
            'diversity': np.mean(diversity_score(cands)),
            'max_rouge1': stat_df.rouge1.max(),
            'avg_rouge1': stat_df.rouge1.mean(),
            'avg_sents': stat_df.num_sents.mean(),
            'avg_toks': stat_df.num_toks.mean()
        })

    agg_stats = pd.DataFrame(agg_stats)
    print(agg_stats.mean())