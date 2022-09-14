import itertools

import pandas as pd


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


if __name__ == '__main__':
    # 'gen_extract_full_ar_mask_red_feat' 'gen_abstract_full' # Gen abstract full
    experiment = 'gen_extract_full_ar_mask_red_feat'
    output = 'validation_sample_outputs'  # validation_from_sample_extract
    summary_style = 'extract'  # from_extract
    col = f'{summary_style}_rouges'

    df = pd.read_csv(f'/nlp/projects/faithsum/results/{experiment}/{output}.csv').dropna(subset=[col])

    extract_rouges = [get_arr(x) for x in df[col].tolist()]
    beam_scores = [get_arr(x) for x in df['extract_beam_scores'].tolist()]
    x = list(itertools.chain(*beam_scores))
    y = list(itertools.chain(*extract_rouges))
    from scipy.stats import pearsonr
    corel = pearsonr(x, y)[0]
    print(corel)
    n = len(df)
    scores = [0 for _ in range(len(extract_rouges[0]))]
    for rouge_arr in extract_rouges:
        for i, s in enumerate(rouge_arr):
            scores[i] += s
    for beam, score in enumerate(scores):
        # print(score / n)
        print(f'Beam {beam}: {score / n}')

    nucleus = []
    from scipy.special import softmax
    import numpy as np
    # for rouge, beam in zip(extract_rouges, beam_scores):
    #     print(beam[0], rouge[0] == max(rouge))
        # dist = softmax(beam)
        # cdf = np.cumsum(dist)
        # print(cdf[:6])
