import os

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def diversity_score(candidates):
    overlaps = []
    for i in range(len(candidates)):
        i_toks = set(list(map(lambda x: x.lower(), candidates[i].split(' '))))
        for j in range(i + 1, len(candidates)):
            j_toks = set(list(map(lambda x: x.lower(), candidates[j].split(' '))))
            avg_len = max(1., 0.5 * len(i_toks) + 0.5 * len(j_toks))
            overlap = len(i_toks.intersection(j_toks)) / avg_len
            overlaps.append(overlap)
    avg_overlap = np.mean(overlaps)
    return 1 - avg_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Score for Diversity')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='extract_indicators')
    parser.add_argument('--experiment', default='gen_abstract_full')
    parser.add_argument('--fn', default='validation_sample_outputs')
    parser.add_argument('--candidate_column', default='abstract')

    args = parser.parse_args()

    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    in_fn = os.path.join(results_dir, args.fn + '.csv')
    data_df = pd.read_csv(in_fn)
    candidates = data_df[args.candidate_column].tolist()

    scores = []
    for candidate_set in tqdm(candidates):
        cand_list = candidate_set.split('<cand>')
        score = diversity_score(cand_list)
        scores.append(score)

    avg_diversity = np.mean(scores)
    print(f'Average Diversity: {avg_diversity}')
