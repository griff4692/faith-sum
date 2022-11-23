import os

import pandas as pd

from datasets import load_from_disk
import argparse


def render(row):
    print('Source: ', row['source_annotated'])
    print('\n')
    print('Reference: ', row.get('summary', row.get('highlights')))
    print('Oracle ROUGE: ', row['oracle_rouge1'])
    print('Oracle focus: ', row['oracle_focus_score'])
    print(row['oracle_idxs'])
    print('\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='xsum_pegasus')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')

    args = parser.parse_args()
    in_dir = os.path.join(args.data_dir, args.dataset)

    dataset = load_from_disk(in_dir)['validation']
    stat_cols = [
        'source_bs_f1',
        'oracle_bs_f1',
        'non_oracle_bs_f1',
        'oracle_focus_score',
        'oracle_rouge1'
    ]

    stat_df = pd.DataFrame({
        col: dataset[col] for col in stat_cols
    })

    corr_df = stat_df.corr(method='pearson')
    corr_df.to_csv(f'./{args.dataset}_corr.csv', index=False)

    for col in stat_cols:
        print(f'{col} {stat_df[col].dropna().mean()}')

    fs = dataset['oracle_focus_score']
    import numpy as np
    focus_order = np.argsort(fs)

    low_focus = focus_order[:5]

    high_focus = focus_order[-5:]
    print('LOW FOCUS')
    for idx in low_focus:
        row = dataset.select([idx])[0]
        render(row)
    print('HIGH FOCUS')
    for idx in high_focus:
        row = dataset.select([idx])[0]
        render(row)

    print('RANDOM\n')
    for idx in np.random.randint(0, len(dataset), size=(10, )):
        row = dataset.select([idx])[0]
        render(row)
