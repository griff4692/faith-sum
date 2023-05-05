import pandas as pd
from scipy.stats import pearsonr
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/faithsum/results/cnn_e_v1/test_from_beam_16_extract_cnn_ea_rand_v2.csv')


    records = df.to_dict('records')

    adhere_scores = []
    non_scores = []
    plan_f1 = []
    x, y = [], []
    by_beam = [[] for _ in range(16)]
    for record in records:
        extract_idx = record['extract_idx'].split('<cand>')
        implied_extract_idx = record['implied_extract_idx'].split('<cand>')
        extract_rouge = record['eval_extract_mean_f1'].split(',')

        vals = record['implied_official_rouge1_f1'].split(',')
        for beam, val in enumerate(vals):
            by_beam[beam].append(float(val))

        for a, b, score in zip(extract_idx, implied_extract_idx, extract_rouge):
            overlaps = len(set(a.split(',')).intersection(set(b.split(','))))
            x.append(overlaps / (0.5 * len(a) + 0.5 * len(b)))
            y.append(float(score))
            if a == b:
                adhere_scores.append(float(score))
            else:
                non_scores.append(float(score))
    print(np.mean(non_scores))
    print(np.mean(adhere_scores))
    print(pearsonr(x, y))

    for beam_arr in by_beam:
        print(np.mean(beam_arr))

