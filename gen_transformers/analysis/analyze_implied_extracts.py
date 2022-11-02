import os
import regex as re

import pandas as pd
import numpy as np
from datasets import load_from_disk


if __name__ == '__main__':
    dataset = 'cnn_dailymail'
    data_dir = '/nlp/projects/faithsum'
    experiment = 'bart_large_cnn'
    summary_type = 'abstract'
    fn = 'test_diverse_16_outputs.csv'

    prefix = ''
    if summary_type == 'abstract':
        prefix = 'implied_'
    idx_col = prefix + 'extract_idx'
    score_col = prefix + 'extract_rouges'

    df = pd.read_csv(os.path.join(data_dir, f'results/{experiment}/{fn}'))
    data_dir = os.path.join(data_dir, dataset)
    dataset = load_from_disk(data_dir)
    beam_num_unique_plans = [[] for _ in range(16)]
    beam_num_unique_extracts = [[] for _ in range(16)]
    beam_implied_rouges = [[] for _ in range(16)]
    source_annotated = dataset['validation']['source_annotated']
    for record in df.to_dict('records'):
        extract_idx = record[idx_col]
        cands = extract_idx.split('<cand>')
        source = source_annotated[record['dataset_idx']]
        num_sents = len(re.findall(r'<s\d+>', source))
        cand_str = []
        try:
            for cand in cands:
                cand_str.append(list(sorted([int(x) for x in cand.split(',')])))
        except:
            print('Empty extract')
            continue
        seen = set()
        seen_extracts = set()
        for beam, arr in enumerate(cand_str):
            arr_str = '_'.join([str(x) for x in arr])
            seen_extracts.add(arr_str)
            for idx in arr:
                seen.add(idx)
            beam_num_unique_plans[beam].append(len(seen) / num_sents)
            beam_num_unique_extracts[beam].append(len(seen_extracts))
        extract_rouges = [float(x) for x in record[score_col].split(',')]
        for beam in range(len(extract_rouges)):
            val = np.mean([extract_rouges[i] for i in range(beam + 1)])
            beam_implied_rouges[beam].append(val)
    print('Fraction Sentences Covered...')
    for beam, arr in enumerate(beam_num_unique_plans):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\nFraction Unique Extracts...')
    for beam, arr in enumerate(beam_num_unique_extracts):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\nExtract ROUGES...')
    for beam, arr in enumerate(beam_implied_rouges):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))
