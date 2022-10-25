import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


if __name__ == '__main__':
    experiment = 'add_doc_bart_large_cnn'
    output = 'test_from_beam_16_extract'
    output = 'test_from_beam_16_extract_bart_large_cnn_from_extract_e3'
    summary_style = 'from_extract_abstract'
    if summary_style == 'abstract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'diversity'
    elif summary_style == 'implied_extract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'implied_diversity'
    elif summary_style == 'from_extract_abstract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'diversity'
    else:
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = f'{summary_style}_diversity'
    score_col = 'rank_scores'
    reranked = False
    print(summary_style)
    in_fn = f'/nlp/projects/faithsum/results/{experiment}/{output}.csv'
    print(in_fn)
    df = pd.read_csv(in_fn).dropna(subset=[summary_style])

    rouges = [get_arr(x) for x in df[rouge_col].tolist()]

    # bartscore_col = f'{summary_style}_bartscores'
    # bartscores = [get_arr(x) for x in df[bartscore_col].tolist()]
    try:
        diversities = [float(x) for x in df[diversity_col]]
    except:
        print('This has been fixed in generate but lets deal with it here.')
        diversities = []
        for x in df[diversity_col]:
            diversities.append(np.mean(json.loads(x)))
    n = len(df)

    rank_scores = None
    if reranked:
        rank_scores = [get_arr(x) for x in df[score_col].tolist()]

    avg_rouges = []
    max_rouges = []
    # avg_bartscores = []
    max_rouges_by_beam = [[] for _ in range(len(rouges[0]))]
    avg_rouges_by_beam = [[] for _ in range(len(rouges[0]))]
    # avg_bartscores_by_beam = [[] for _ in range(len(rouges[0]))]

    for i in range(n):
        rouge_arr = rouges[i]
        # bartscore_arr = bartscores[i]
        rouge_arr_sorted = rouge_arr
        # bartscore_arr_sorted = bartscore_arr
        if rank_scores is not None:
            scores = rank_scores[i]
            priority = np.argsort(-np.array(scores))
            rouge_arr_sorted = [rouge_arr[pidx] for pidx in priority]
            # bartscore_arr_sorted = [bartscore_arr[pidx] for pidx in priority]

        avg_rouges.append(np.mean(rouge_arr))
        # avg_bartscores.append(np.mean(bartscore_arr_sorted))
        max_rouges.append(max(rouge_arr))
        for beam in range(len(avg_rouges_by_beam)):
            cum_rouge = rouge_arr_sorted[:beam + 1]
            avg_rouges_by_beam[beam].append(np.mean(cum_rouge))
            max_rouges_by_beam[beam].append(max(cum_rouge))
            # cum_bartscore = bartscore_arr_sorted[:beam + 1]
            # avg_bartscores_by_beam[beam].append(np.mean(cum_bartscore))

    print(f'Mean Avg inverse SELF-BLEU: {np.mean(diversities)}')
    print(f'Mean Avg ROUGE-1 F1: {np.mean(avg_rouges)}')
    print(f'Mean Max ROUGE-1 F1: {np.mean(max_rouges)}')
    # print(f'Mean Avg BartScore: {np.mean(avg_bartscores)}')
    print('Mean Avg ROUGE-1 F1 by Beam...')
    out = []
    for beam in range(len(avg_rouges_by_beam)):
        out.append(str(np.mean(avg_rouges_by_beam[beam])))
    print('\t'.join(out))

    print('Mean Max ROUGE-1 F1 by Beam...')
    out = []
    for beam in range(len(max_rouges_by_beam)):
        out.append(str(np.mean(max_rouges_by_beam[beam])))
    print('\t'.join(out))

    # print('Mean Cumulative BartScore by Beam...')
    # out = []
    # for beam in range(len(avg_rouges_by_beam)):
    #     out.append(str(np.mean(avg_bartscores_by_beam[beam])))
    # print('\t'.join(out))
