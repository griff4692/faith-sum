import itertools

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
    # 'gen_extract_full_ar_mask_red_feat' 'gen_abstract_full' # Gen abstract full
    experiment = 'gen_extract_full_ar_mask_red_feat'
    output = 'validation_from_sample_extract'  # validation_from_sample_extract
    summary_style = 'extract'  # from_extract
    col = f'{summary_style}_rouges'
    df = pd.read_csv(f'/nlp/projects/faithsum/results/{experiment}/{output}.csv').dropna(subset=[col])

    rouges = [get_arr(x) for x in df[col].tolist()]

    records = df.to_dict('records')
    for record in records:
        extracts = record['extract'].split('<cand>')
        from_extracts = record['from_extract_abstract'].split('<cand>')
        extract_rouges = get_arr(record['extract_rouges'])
        from_extract_rouges = get_arr(record['from_extract_rouges'])
        beam = 1
        print('Source: ', record['source'], '\n')
        print('Reference: ', record['reference'], '\n')
        for extract, from_extracts, er, fer in zip(extracts, from_extracts, extract_rouges, from_extract_rouges):
            print('Beam ' + str(beam) + '\n')
            print('Extract: ROUGE ', er)
            print(extract)
            print('From Extract: ROUGE ', fer)
            print(from_extracts)
            beam += 1
            print('\n')
        print('\n\n\n')

    if 'extract_beam_scores' in df:
        beam_scores = [get_arr(x) for x in df['extract_beam_scores'].tolist()]
        corels = []
        top_ranks = []
        for bs, er in zip(beam_scores, rouges):
            corel = spearmanr(bs, er)[0]
            if np.isnan(corel):
                continue
            corels.append(corel)
            top_ranks.append(int(np.argmax(er) + 1))
        avg_corel = str(round(float(np.mean(corels)), 2))
        print(f'Average Correlation Between Beam Score and ROUGE: {avg_corel}')
    else:
        top_ranks = []
        for er in rouges:
            top_ranks.append(int(np.argmax(er) + 1))
    avg_rank = str(round(float(np.mean(top_ranks)), 2))
    print(f'Average Beam Rank of Highest ROUGE summary: {avg_rank}')

    # exit(0)

    n = len(df)
    scores = [0 for _ in range(len(rouges[0]))]
    for rouge_arr in rouges:
        for i, s in enumerate(rouge_arr):
            scores[i] += s
    for beam, score in enumerate(scores):
        # print(score / n)
        print(f'Beam {beam + 1}: {score / n}')

    nucleus = []
    from scipy.special import softmax
    # for rouge, beam in zip(extract_rouges, beam_scores):
    #     print(beam[0], rouge[0] == max(rouge))
        # dist = softmax(beam)
        # cdf = np.cumsum(dist)
        # print(cdf[:6])
