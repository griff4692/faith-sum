import itertools
import os
import regex as re

import argparse
import spacy
import numpy as np
from datasets import load_from_disk
from bert_score.scorer import BERTScorer

from preprocess.extract_oracles import convert_to_sents
from sum_constants import summarization_name_mapping


def _add_bert_alignment(bs, source_sents, target_sents, threshold=0.9):
    added_sents = []
    all_sents = set()
    for target_sent in target_sents:
        target_rep = [target_sent for _ in range(len(source_sents))]
        scores = bs.score(target_rep, source_sents)
        # p = scores[0]
        # p_adj = [0 if i in all_sents else x for i, x in enumerate(p.tolist())]
        # r = scores[1]
        # r_adj = [0 if i in all_sents else x for i, x in enumerate(r.tolist())]
        f1 = scores[-1]
        f1_adj = [0 if i in all_sents else x for i, x in enumerate(f1.tolist())]
        keep_sents = [i for i in range(len(f1)) if f1[i] >= threshold]
        if len(keep_sents) == 0:
            # None over threshold take argmax
            keep_sents = [int(np.argmax(f1_adj))]
        for sent_idx in keep_sents:
            all_sents.add(sent_idx)
        added_sents.append(keep_sents)
    oracle_idxs = list(sorted(list(set(list(itertools.chain(*added_sents))))))
    return oracle_idxs, added_sents


def add_bert_alignment(nlp, batch_data, target_col, bs):
    target = batch_data[target_col]
    target_sents = list(map(str, convert_to_sents(target, nlp, is_dialogue=False)))
    source_sents = []
    tps = re.split(r'(<s\d+>)', batch_data['source_annotated'])
    for tp_idx, tp in enumerate(tps):
        if re.match(r'(<s\d+>)', tp) is not None:
            source_sents.append(tps[tp_idx + 1].strip())

    bert_idxs, bert_alignments = _add_bert_alignment(bs, source_sents, target_sents)

    rouge_oracle = list(sorted(batch_data['oracle_idxs']))
    if bert_idxs == rouge_oracle:
        print('equal')
    else:
        print('\n', bert_idxs, rouge_oracle)
        print(bert_alignments)
        print('\n')
        print(batch_data['source_annotated'])
        print('\n')
        print('Summary: ', target)
        print('\n\n\n')
        print('not equal')

    return {
        'bert_idxs': bert_idxs,
        'bert_alignments': bert_alignments
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--splits', default='validation')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--device', default=1, type=int)

    args = parser.parse_args()

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')
    bs = BERTScorer(lang='en')
    input_col, target_col = summarization_name_mapping[args.dataset]
    out_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(out_dir)
    encoded_data = {}
    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        encoded = dataset[split].map(lambda examples: add_bert_alignment(
            nlp, examples, target_col, bs
        ), batched=False, num_proc=1)
        encoded = encoded.filter(lambda example: len(example[input_col].strip()) > 0)
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
