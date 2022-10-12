from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import argparse

from eval.bart_score import BARTScorer


DEFAULT_FN = '/nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_from_beam_1_extract.csv'


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


def evaluate_summary(rouge_metric, generated, gold, prefix=''):
    outputs = rouge_metric.evaluate_batch([generated], [gold], aggregate=True)['rouge']
    stats = {}
    for rouge_type in ['1', '2', 'L']:
        fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
        stats[f'{prefix}rouge{rouge_type}_precision'] = outputs[f'rouge_{rouge_type.lower()}_precision']
        stats[f'{prefix}rouge{rouge_type}_recall'] = outputs[f'rouge_{rouge_type.lower()}_recall']
        stats[f'{prefix}rouge{rouge_type}_f1'] = fscore
    return stats


def process_example(args, record, bart_scorer):
    summaries = record[args.column].split('<cand>')
    n = len(summaries)
    source = record['source']
    source_rep = [source for _ in range(n)]
    scores = bart_scorer.score(source_rep, summaries, batch_size=4)
    metric_str = ','.join(list(map(str, scores)))
    record[args.column + '_bartscores'] = metric_str
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD BartScore')

    parser.add_argument('--experiment', default='add_doc')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--fn', default=DEFAULT_FN)
    parser.add_argument('--split', default='test')
    parser.add_argument('--column', default='from_extract_abstract', choices=[
        'from_extract_abstract', 'abstract',
    ])
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    bart_scorer = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn')

    outputs = pd.read_csv(args.fn)
    records = outputs.to_dict('records')

    augmented_records = list(tqdm(map(
        lambda record: process_example(args, record, bart_scorer), records), total=len(records)
    ))
    augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

    print(f'Saving with BartSCore added back to {args.fn}')
    augmented_df.to_csv(args.fn, index=False)
