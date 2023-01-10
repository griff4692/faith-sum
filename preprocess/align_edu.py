import os
import regex as re

import argparse
import numpy as np
from datasets import load_from_disk

from gen_transformers.model_utils import infer_hf_model
from preprocess.convert_abstractive_to_extractive import _calc_rouge


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


def align_example_edus(batch):
    oracle_idxs = []
    oracle_alignments = []
    oracle_soft_labels = []
    for source_annot, target_annot, max_num in zip(
            batch['source_annotated_edu'], batch['target_annotated_edu'], batch['num_edus_post_trunc']
    ):
        source_edus = edus_from_html(source_annot)[:max_num]
        target_edus = edus_from_html(target_annot)

        score_matrix = align_edus(source_edus, target_edus)
        osl = list(map(float, np.max(score_matrix, axis=1).tolist()))
        oa = list(map(int, np.argmax(score_matrix, axis=0).tolist()))

        oracle_alignments.append(oa)
        oracle_idxs.append(list(sorted(list(set(oa)))))
        oracle_soft_labels.append(osl)

    return {
        'oracle_alignments': oracle_alignments,
        'oracle_idxs': oracle_idxs,
        'oracle_soft_labels': oracle_soft_labels
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Oracle Align Target EDUs to Source EDUs to the EDU-level datasets.')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--num_proc', default=64, type=int)

    args = parser.parse_args()

    infer_hf_model(args, is_abstract=False)

    out_dir = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    print(f'Saving to {out_dir}')

    print(f'Loading {args.dataset}...')
    edu_dir = os.path.join(args.data_dir, args.dataset + '_edus')
    dataset = load_from_disk(edu_dir)

    encoded_data = {}
    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        encoded = dataset[split].map(
            align_example_edus,
            batched=True, batch_size=1000, num_proc=args.num_proc,
        )
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
