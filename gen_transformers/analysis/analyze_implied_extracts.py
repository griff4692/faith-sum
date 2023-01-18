import os
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk


from gen_transformers.data_utils import infer_dataset
from preprocess.align_edu import edus_from_html


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze implied EDU extracts.')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--experiment', default='bart_large_cnn')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--fn', default='test_nucleus_16_outputs.csv')

    args = parser.parse_args()

    infer_dataset(args, 'experiment')

    idx_col = 'implied_extract_idx'
    score_col = 'implied_official_rouge1_f1'

    df = pd.read_csv(os.path.join(args.data_dir, f'results/{args.experiment}/{args.fn}'))
    data_dir = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    print(f'Loading dataset from {data_dir}')
    dataset = load_from_disk(data_dir)
    beam_num_unique_plans = [[] for _ in range(16)]
    beam_num_unique_extracts = [[] for _ in range(16)]
    beam_implied_rouges = [[] for _ in range(16)]
    beam_implied_rouges_cum = [[] for _ in range(16)]
    beam_implied_rouges_max = [[] for _ in range(16)]
    source_annotated = dataset['test']['source_edu_annotated']
    n = len(df)
    for record in tqdm(df.to_dict('records'), total=n):
        extract_idx = record[idx_col]
        cands = extract_idx.split('<cand>')
        source = source_annotated[record['dataset_idx']]
        sedus = edus_from_html(source)
        num_sents = len(sedus)
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
            beam_implied_rouges_cum[beam].append(val)
            beam_implied_rouges_max[beam].append(float(max([extract_rouges[i] for i in range(beam + 1)])))
            beam_implied_rouges[beam].append(extract_rouges[beam])
    print('Fraction Sentences Covered...')
    for beam, arr in enumerate(beam_num_unique_plans):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\n\# of Unique Extracts...')
    for beam, arr in enumerate(beam_num_unique_extracts):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\nCumulative Avg ROUGES @ Each Beam...')
    for beam, arr in enumerate(beam_implied_rouges_cum):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\nCumulative Max ROUGES...')
    for beam, arr in enumerate(beam_implied_rouges_max):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))

    print('\nExtract ROUGES @ Each Beam...')
    for beam, arr in enumerate(beam_implied_rouges):
        # print(f'{beam + 1},{np.mean(arr)}')
        print(np.mean(arr))
