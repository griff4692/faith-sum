from bert_score.scorer import BERTScorer
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np


ea_fn = '/nlp/projects/faithsum/results/samsum_extract_generator/test_from_beam_16_extract.csv'
diverse_fn = '/nlp/projects/faithsum/results/test_diverse_16_outputs.csv'
beam_fn = '/nlp/projects/faithsum/results/bart_large_samsum/test_beam_16_outputs.csv'
DEFAULT_FN = ea_fn


def process_example(record, bs):
    if 'from_extract_abstract' in record:
        summaries = record['from_extract_abstract'].split('<cand>')
    else:
        summaries = record['abstract'].split('<cand>')
    n = len(summaries)
    source = record['source']
    source_rep = [source for _ in range(n)]
    bs.score(target_rep, source_sents)
    beam_scores = bs.score(source_rep, summaries)['scores']
    metric_str = ','.join(list(map(str, beam_scores)))
    record['summac'] = metric_str
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD BartScore')

    parser.add_argument('--experiment', default='add_doc')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--fn', default=DEFAULT_FN)
    parser.add_argument('--column', default='from_extract_abstract', choices=[
        'from_extract_abstract', 'abstract',
    ])
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    bs = BERTScorer(lang='en')

    print(f'Reading in records from {args.fn}')
    outputs = pd.read_csv(args.fn)
    # outputs = outputs.sample(n=100, replace=False)
    records = outputs.to_dict('records')

    augmented_records = list(tqdm(map(
        lambda record: process_example(record, model_conv), records), total=len(records)
    ))
    augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

    print(f'Saving with SummaC added back to {args.fn}')
    augmented_df.to_csv(args.fn, index=False)

    scores_by_beam = [[] for _ in range(16)]

    for record in augmented_records:
        v = list(map(float, record['summac'].split(',')))
        for beam, v in enumerate(v):
            scores_by_beam[beam].append(v)

    for beam in range(len(scores_by_beam)):
        print(str(beam) + ' ' + str(np.mean(scores_by_beam[beam])))
