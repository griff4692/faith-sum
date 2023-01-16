import os
from summac.model_summac import SummaCConv

import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np


FNS_BY_DATASET = {
    'cnn_dailymail': [
        'bart_large_cnn/test_diverse_16_outputs',
        'bart_large_cnn/test_nucleus_16_outputs',
        'bart_large_cnn/test_beam_16_outputs',
        'cnn_e_v1/test_from_beam_16_extract_cnn_ea_rand',
    ],
    'xsum': [
        'pegasus_xsum/test_diverse_16_outputs_1dp.csv',
        'pegasus_xsum/test_beam_16_outputs.csv',
        'test_nucleus_16_outputs',
    ]
}


def process_example(record, model_conv):
    if 'from_extract_abstract' in record:
        summaries = record['from_extract_abstract'].split('<cand>')
    else:
        summaries = record['abstract'].split('<cand>')
    n = len(summaries)
    source = record['source']
    source_rep = [source for _ in range(n)]
    beam_scores = model_conv.score(source_rep, summaries)['scores']
    metric_str = ','.join(list(map(str, beam_scores)))
    record['summac'] = metric_str
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add SummaC')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum/results')
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    # https://github.com/tingofurro/summac/
    model_conv = SummaCConv(
        models=['vitc'], bins='percentile', granularity="sentence", nli_labels='e', device=args.device,
        start_file='default', agg='mean',
    )

    fns = FNS_BY_DATASET[args.dataset]
    for fn in fns:
        fn = os.path.join(args.data_dir, fn)
        if not fn.endswith('.csv'):
            fn += '.csv'

        out_fn = fn.split('.')[0] + '_summac.csv'
        if os.path.exists(out_fn):
            print(fn + ' already exists. Skipping. Delete to re-run')
            continue
        else:
            print(f'Will save results to {out_fn}')
        print(f'Reading in records from {fn}')
        outputs = pd.read_csv(fn)
        records = outputs.to_dict('records')

        augmented_records = list(tqdm(map(
            lambda record: process_example(record, model_conv), records), total=len(records)
        ))
        augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

        print(f'Saving with SummaC added back to {out_fn}')
        augmented_df.to_csv(out_fn, index=False)

        scores_by_beam = [[] for _ in range(16)]
        for record in augmented_records:
            v = list(map(float, record['summac'].split(',')))
            for beam, v in enumerate(v):
                scores_by_beam[beam].append(v)

        print(f'SummaC for {fn}')
        for beam in range(len(scores_by_beam)):
            print(str(np.mean(scores_by_beam[beam])))

