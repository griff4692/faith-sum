from collections import defaultdict
from datasets import load_from_disk

import pandas as pd
import argparse
import os
from collections import Counter
from p_tqdm import p_uimap
import regex as re

from eval.rouge_metric import RougeMetric


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


def find_article(brio_fn, brio_obj, articles_no_space, articles_vocab, covered_ids):
    brio_article_no_space = re.sub(r'[^a-z]+', '', ''.join(brio_obj['article'])).lower().strip()
    found_idxs = []
    vocab_counts = Counter(' '.join(brio_obj['article']).lower().strip().split(' '))
    for cand_idx, ans in enumerate(articles_no_space):
        if ids[cand_idx] in covered_ids:
            continue
        if brio_article_no_space in ans or ans in brio_article_no_space:
            found_idxs.append(cand_idx)
            break

    if len(found_idxs) == 1:
        best_cand_idx = found_idxs[0]
    else:
        best_match = 0
        best_cand_idx = None
        for cand_idx, vocab in enumerate(articles_vocab):
            if ids[cand_idx] in covered_ids:
                continue
            match = 0
            for k, v in vocab_counts.items():
                if k in vocab:
                    match += vocab[k]
            if match > best_match:
                best_match = match
                best_cand_idx = cand_idx
    return brio_fn, best_cand_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD PERL ROUGE Eval')

    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    args = parser.parse_args()

    rouge_metric = RougeMetric()

    data_dir = os.path.join(args.data_dir, 'cnndm_cased')

    from glob import glob
    pattern = os.path.join(data_dir, '*', '*.json')

    fns = list(glob(pattern))

    print(len(fns))

    orig_data_dir = os.path.join(args.data_dir, 'cnn_dailymail')
    dataset = load_from_disk(orig_data_dir)
    import ujson
    from tqdm import tqdm
    brio_inputs = {'validation': {}, 'test': {}, 'train': {}}
    for fn in tqdm(fns):
        with open(fn, 'r') as fd:
            brio_output = ujson.load(fd)
            if 'train' in fn:
                split = 'train'
            elif 'val' in fn:
                split = 'validation'
            else:
                split = 'test'
            brio_inputs[split][fn] = brio_output

    brio_outputs = {
        'train': {},
        'validation': {},
        'test': {},
    }

    for split, split_data in dataset.items():
        articles = split_data['article']
        split_brio = brio_inputs[split]

        articles_vocab = []
        for article in tqdm(articles):
            articles_vocab.append(Counter(article.lower().strip().split(' ')))

        articles_no_space = list(tqdm(map(
            lambda x: re.sub(r'[^a-z]+', '', x).lower().strip(), articles), total=len(articles)))
        ids = split_data['id']
        covered_ids = set()

        outputs = list(p_uimap(
            lambda fn: find_article(fn, split_brio[fn], articles_no_space, articles_vocab, covered_ids), split_brio))

        for fn, best_cand_idx in outputs:
            candidates = []
            cand_rouges = []
            for cand in brio_inputs[split][fn]:
                candidates.append('\n'.join(cand[0]))
                cand_rouges.append(float(cand[1]))
            brio_outputs[split][ids[best_cand_idx]] = [candidates, cand_rouges, fn, best_cand_idx]
            # covered_ids.add(ids[best_cand_idx])

    out_fn = os.path.join(args.data_dir, 'cnn_brio_outputs.json')
    print(f'Saving data to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(brio_outputs, fd)

    exit(0)
    augmented_records = p_uimap(lambda record: process_example(args, record, rouge_metric), records, num_cpus=16)
    # augmented_records = list(map(lambda record: process_example(args, record, rouge_metric), records))
    augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)

    # print(f'Saving with PERL eval ROUGE columns added back to {out_fn}')
    # augmented_df.to_csv(out_fn, index=False)
