import os
import regex as re

import pandas as pd
import numpy as np
from datasets import load_from_disk


if __name__ == '__main__':
    data_dir = '/nlp/projects/faithsum'
    experiment = 'gen_abstract_full'  # 'gen_abstract_full' gen_extract_full_ar_mask_red_feat
    summary_type = 'abstract'

    prefix = ''
    if summary_type == 'abstract':
        prefix = 'implied_'
    idx_col = prefix + 'extract_idx'
    score_col = prefix + 'extract_rouges'

    df = pd.read_csv(os.path.join(data_dir, f'results/{experiment}/validation_sample_outputs.csv'))
    data_dir = os.path.join(data_dir, 'cnn_dailymail')
    dataset = load_from_disk(data_dir)
    implied_idxs = df['implied_extract_idx'].tolist()
    beam_num_unique_plans = [[] for _ in range(16)]
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
        for beam, arr in enumerate(cand_str):
            for idx in arr:
                seen.add(idx)
            beam_num_unique_plans[beam].append(len(seen) / num_sents)
        extract_rouges = [float(x) for x in record[score_col].split(',')]
        for beam in range(len(extract_rouges)):
            val = np.mean([extract_rouges[i] for i in range(beam + 1)])
            beam_implied_rouges[beam].append(val)
    print('Fraction Sentences Covered...')
    for beam, arr in enumerate(beam_num_unique_plans):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))
    print('Extract ROUGES...')
    for beam, arr in enumerate(beam_implied_rouges):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))
