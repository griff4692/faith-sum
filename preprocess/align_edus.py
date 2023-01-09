import json
import os
import regex as re

import argparse
import numpy as np
from datasets import load_from_disk
import pandas as pd

from preprocess.convert_abstractive_to_extractive import _calc_rouge


DATA_DIR = '/nlp/projects/faithsum'


def edus_from_html(text):
    edus = []
    tps = re.split(r'(</?e>)', text)
    for i, tp in enumerate(tps):
        if tp == '<e>':
            assert tps[i + 2] == '</e>'
            edus.append(tps[i + 1])
    return edus


def score_edu_pair(a, b):
    obj = _calc_rouge([[a]], [[b]])
    return (obj['rouge_1'] + obj['rouge_2']) / 2.0


def align_edus(sedus, tedus):
    scores = np.zeros([len(sedus), len(tedus)])
    for s_idx, sedu in enumerate(sedus):
        for t_idx, tedu in enumerate(tedus):
            score = score_edu_pair(sedu, tedu)
            scores[s_idx, t_idx] = score
    return scores


if __name__ == '__main__':
    in_dir = os.path.join(DATA_DIR, 'cnn_dailymail_sentences')

    with open('cnn_example_edu.json', 'r') as fd:
        edus = json.load(fd)
        sedu = [x for x in edus if x['type'] == 'source']
        tedu = [x for x in edus if x['type'] == 'target']

        source_sents_w_edu = list(sorted(sedu, key=lambda x: x['sent_idx']))
        target_sents_w_edu = list(sorted(tedu, key=lambda x: x['sent_idx']))

        flat_source_sents_w_edu = ' '.join(list(map(lambda x: x['sent_w_edu'], source_sents_w_edu)))
        flat_target_sents_w_edu = ' '.join(list(map(lambda x: x['sent_w_edu'], target_sents_w_edu)))

        source_edus = edus_from_html(flat_source_sents_w_edu)
        target_edus = edus_from_html(flat_target_sents_w_edu)

        score_matrix = align_edus(source_edus, target_edus)
        oracle_alignments = list(map(int, np.argmax(score_matrix, axis=0).tolist()))

        for idx in range(len(oracle_alignments)):
            print(target_edus[idx] + ' -> ' + source_edus[oracle_alignments[idx]])

    dataset = load_from_disk(in_dir)

    val = dataset['validation']
    ids = val['id']
    ex_idx = ids.index('330e4546c7aac748dd5900e7f5811ddbe45abd8e')

    example = val.select([ex_idx])[0]
