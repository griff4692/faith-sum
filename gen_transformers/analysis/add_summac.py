from summac.model_summac import SummaCConv
from bert_score.scorer import BERTScorer
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np


ea_fn = '/nlp/projects/faithsum/results/samsum_extract_generator/test_from_beam_16_extract.csv'
diverse_fn = '/nlp/projects/faithsum/results/bart_large_samsum/test_diverse_16_outputs.csv'
beam_fn = '/nlp/projects/faithsum/results/bart_large_samsum/test_beam_16_outputs.csv'
DEFAULT_FNS = ea_fn + ',' + diverse_fn + ',' + beam_fn


def process_example(record, model_conv, bs):
    if 'from_extract_abstract' in record:
        summaries = record['from_extract_abstract'].split('<cand>')
    else:
        summaries = record['abstract'].split('<cand>')
    n = len(summaries)
    source = record['source']
    source_rep = [source for _ in range(n)]
    beam_scores = model_conv.score(source_rep, summaries)['scores']
    bert_p, bert_r, bert_f1 = bs.score(cands=summaries, refs=source_rep)
    metric_str = ','.join(list(map(str, beam_scores)))
    record['summac'] = metric_str
    record['bertscore_p'] = ','.join(list(map(str, bert_p.cpu().numpy())))
    record['bertscore_r'] = ','.join(list(map(str, bert_r.cpu().numpy())))
    record['bertscore_f1'] = ','.join(list(map(str, bert_f1.cpu().numpy())))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD BartScore')

    parser.add_argument('--experiment', default='add_doc')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--fns', default=DEFAULT_FNS)
    parser.add_argument('--column', default='from_extract_abstract', choices=[
        'from_extract_abstract', 'abstract',
    ])
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    # https://github.com/tingofurro/summac/
    model_conv = SummaCConv(
        models=['vitc'], bins='percentile', granularity="sentence", nli_labels='e', device=args.device, start_file='default',
        agg='mean'
    )

    bs = BERTScorer(lang='en')

    for fn in args.fns.split(','):
        print(f'Reading in records from {fn}')
        outputs = pd.read_csv(fn)
        records = outputs.to_dict('records')

        augmented_records = list(tqdm(map(
            lambda record: process_example(record, model_conv, bs), records), total=len(records)
        ))
        augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

        print(f'Saving with SummaC/BertScore added back to {fn}')
        augmented_df.to_csv(fn, index=False)

        scores_by_beam = [[] for _ in range(16)]
        bs_by_beam = [[] for _ in range(16)]
        for record in augmented_records:
            v = list(map(float, record['summac'].split(',')))
            b = list(map(float, record['bertscore_p'].split(',')))
            for beam, v in enumerate(v):
                scores_by_beam[beam].append(v)
                bs_by_beam[beam].append(b[beam])

        print(f'SummaC for {fn}')
        for beam in range(len(scores_by_beam)):
            print(str(np.mean(scores_by_beam[beam])))

        print(f'BertScore Source Precision for {fn}')
        for beam in range(len(scores_by_beam)):
            print(str(np.mean(bs_by_beam[beam])))
