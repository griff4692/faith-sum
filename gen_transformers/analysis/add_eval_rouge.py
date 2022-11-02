import os
from collections import defaultdict

import pandas as pd
import argparse
from p_tqdm import p_uimap

from eval.rouge_metric import RougeMetric


DEFAULT_FN = '/nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_from_beam_16_extract.csv'


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


def process_example(args, record, rouge_metric):
    try:
        metric_str = defaultdict(list)
        for col in args.columns.split(','):
            summaries = record[col].split('<cand>')
            metrics = list(map(lambda s: evaluate_summary(rouge_metric, s, record['reference']), summaries))
            mean_f1s = []
            for m in metrics:
                for k, v in m.items():
                    metric_str['eval' + '_' + col + '_' + k].append(str(v))
                mean_f1 = (m['rouge1_f1'] + m['rouge2_f1'] + m['rougeL_f1']) / 3.0
                mean_f1s.append(mean_f1)
                metric_str['eval' + '_' + col + '_mean_f1'].append(str(mean_f1))

            # existing_rouges = get_arr(record[col + '_rouges'])
            # corel = spearmanr(existing_rouges, mean_f1s)[0]
        for k in metric_str:
            metric_str[k] = ','.join(metric_str[k])
        record.update(metric_str)
    except Exception as e:
        print('Issue parsing example: ', str(e))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD PERL ROUGE Eval')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum/results')
    parser.add_argument('--fn', default=DEFAULT_FN)
    parser.add_argument('--columns', default='extract,from_extract_abstract', choices=[
        'extract', 'extract,from_extract_abstract', 'abstract', 'abstract,implied_extract'
    ])
    parser.add_argument('--num_cpus', default=32, type=int)

    args = parser.parse_args()
    rouge_metric = RougeMetric()

    out_fn = os.path.join(args.data_dir, args.fn)
    outputs = pd.read_csv(out_fn)
    records = outputs.to_dict('records')

    args = parser.parse_args()

    augmented_records = p_uimap(
        lambda record: process_example(args, record, rouge_metric), records, num_cpus=args.num_cpus
    )
    # augmented_records = list(map(lambda record: process_example(args, record, rouge_metric), records))
    augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

    print(f'Saving with PERL eval ROUGE columns added back to {args.fn}')
    augmented_df.to_csv(out_fn, index=False)

    for col in augmented_df.columns.tolist():
        if 'rouge' in col and 'eval' in col:
            try:
                print(col, ': ', str(augmented_df[col].dropna().mean()))
            except:
                print(f'Array col: {col}')
