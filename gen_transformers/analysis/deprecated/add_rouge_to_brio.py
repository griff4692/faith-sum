from datasets import load_from_disk
import ujson
from tqdm import tqdm

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


def process_example(brio_outputs, rouge_metric, reference, source):
    candidates = brio_outputs[0]
    metrics = list(map(lambda s: evaluate_summary(rouge_metric, s, reference), candidates))
    brio_outputs.append(metrics)
    print(candidates)
    print('\n\n')
    print(reference)
    print('\n\n')
    print(source)
    print('\n\n\n\n\n\n')
    return 1


def find_article(brio_fn, brio_obj, articles_no_space, articles_vocab):
    brio_article_no_space = re.sub(r'[^a-z]+', '', ''.join(brio_obj['article'])).lower().strip()
    found_idxs = []
    vocab_counts = Counter(' '.join(brio_obj['article']).lower().strip().split(' '))
    for cand_idx, ans in enumerate(articles_no_space):
        if brio_article_no_space in ans or ans in brio_article_no_space:
            found_idxs.append(cand_idx)
            break

    if len(found_idxs) == 1:
        best_cand_idx = found_idxs[0]
    else:
        best_match = 0
        best_cand_idx = None
        for cand_idx, vocab in enumerate(articles_vocab):
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

    out_fn = os.path.join(args.data_dir, 'cnn_brio_outputs.json')

    orig_data_dir = os.path.join(args.data_dir, 'cnn_dailymail')
    print('Loading original data')
    dataset = load_from_disk(orig_data_dir)
    #
    # train_id2reference = dict(zip(dataset['train']['id'], dataset['train']['highlights']))
    # train_id2source = dict(zip(dataset['train']['id'], dataset['train']['article']))
    #
    # with open(out_fn, 'r') as fd:
    #     brio_outputs = ujson.load(fd)
    #
    # for did, arr in brio_outputs['train'].items():
    #     ref = train_id2reference[did]
    #     print(ref)
    #     print(arr[0][0])
    #     print('\n\n\n')
    #
    # exit(0)
    if True:  # not os.path.exists(out_fn):
        data_dir = os.path.join(args.data_dir, 'cnndm_cased')

        from glob import glob
        pattern = os.path.join(data_dir, '*', '*.json')

        fns = list(glob(pattern))[:100]

        print(len(fns))
        brio_inputs = {'validation': {}, 'test': {}, 'train': {}}
        for fn in tqdm(fns):
            with open(fn, 'r') as fd:
                if 'train' in fn:
                    split = 'train'
                elif 'val' in fn:
                    split = 'validation'
                else:
                    split = 'test'
                brio_inputs[split][fn] = ujson.load(fd)

        brio_outputs = {
            'train': {},
            'validation': {},
            'test': {},
        }

        for split in ['validation', 'test', 'train']:
            split_data = dataset[split]
            articles = split_data['article']
            split_ids = split_data['id']
            split_brio = brio_inputs[split]

            articles_vocab = []
            for article in tqdm(articles):
                articles_vocab.append(Counter(article.lower().strip().split(' ')))

            articles_no_space = list(tqdm(map(
                lambda x: re.sub(r'[^a-z]+', '', x).lower().strip(), articles), total=len(articles)))
            outputs = list(p_uimap(
                lambda fn: find_article(
                    fn, split_brio[fn], articles_no_space, articles_vocab), split_brio
            ))

            for fn, best_cand_idx in outputs:
                candidates = []
                cand_rouges = []
                for cand in brio_inputs[split][fn]['candidates']:
                    candidates.append('\n'.join(cand[0]))
                    cand_rouges.append(float(cand[1]))
                brio_outputs[split][split_ids[best_cand_idx]] = [candidates, cand_rouges, fn, best_cand_idx]

        out_fn = os.path.join(args.data_dir, 'cnn_brio_outputs.json')
        print(f'Saving data to {out_fn}')
        with open(out_fn, 'w') as fd:
            ujson.dump(brio_outputs, fd)
    else:
        val_id2reference = dict(zip(dataset['validation']['id'], dataset['validation']['highlights']))
        val_id2source = dict(zip(dataset['validation']['id'], dataset['validation']['article']))
        with open(out_fn, 'r') as fd:
            brio_outputs = ujson.load(fd)
        valid_brio_outputs = brio_outputs['validation']
        complete = sum(list(p_uimap(
            lambda dataset_id: process_example(
                valid_brio_outputs[dataset_id], rouge_metric, reference=val_id2reference[dataset_id],
                source=val_id2source[dataset_id]
            ), valid_brio_outputs, num_cpus=1
        )))
        print(complete)

        # print(f'Saving data to {out_fn}')
        # with open(out_fn, 'w') as fd:
        #     ujson.dump(brio_outputs, fd)
