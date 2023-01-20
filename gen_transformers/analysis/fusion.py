import itertools
import os

import numpy as np
import pandas as pd
import regex as re
from datasets import load_from_disk
import spacy
from p_tqdm import p_uimap

from preprocess.align_edu import edus_from_html
from preprocess.extract_oracles import convert_to_sents


def remove_non_ascii(text):
    return text.encode('ascii', errors='ignore').decode()


def process_oracle(record, nlp, source_edu_annotated, oracle_idxs):
    source = remove_non_ascii(record['source'])
    edus = edus_from_html(source_edu_annotated[record['dataset_idx']])
    edus_no_space = [re.sub(r'\W+', '', edu).lower() for edu in edus]
    source_sents = list(map(str, convert_to_sents(source, nlp)))
    reference = record['reference']
    source_sents_no_space = [re.sub(r'\W+', '', sent).lower() for sent in source_sents]
    ref_sents = list(map(str, convert_to_sents(reference, nlp)))
    num_pred_sents = len(ref_sents)
    frags = [edus_no_space[i] for i in oracle_idxs]
    sent_idx = []
    valid = True
    for frag in frags:
        found = False
        for sidx, sent in enumerate(source_sents_no_space):
            if frag in sent:
                sent_idx.append(sidx)
                found = True
                break
        if not found:
            valid = False
            break
    if valid:
        return {
            'num_implied_edu': len(oracle_idxs),
            'sent_pos': float(np.mean([x + 1 for x in sent_idx])),
            'num_implied_sent': len(set(sent_idx)),
            'num_pred_sents': num_pred_sents,
            'fusion_score': len(set(sent_idx)) / num_pred_sents,
        }
    return None


def process(record, nlp, source_edu_annotated, prefix, beams):
    source = remove_non_ascii(record['source'])
    edus = edus_from_html(source_edu_annotated[record['dataset_idx']])
    edus_no_space = [re.sub(r'\W+', '', edu).lower() for edu in edus]
    source_sents = list(map(str, convert_to_sents(source, nlp)))
    col = 'from_extract_abstract' if 'from_extract_abstract' in record else 'abstract'
    predictions = record[col].split('<cand>')
    source_sents_no_space = [re.sub(r'\W+', '', sent).lower() for sent in source_sents]
    if type(record[f'{prefix}extract_idx']) == float:
        print('Invalid')
        return []
    extract_idx = record[f'{prefix}extract_idx'].split('<cand>')
    try:
        extract_idx = [extract_idx[b] for b in beams]
    except:
        print('Out of range index in extracts...')
        return []
    outs = []
    for cand_idx, ei in enumerate(extract_idx):
        ei = [int(x) for x in ei.split(',')]
        pred_sents = list(map(str, convert_to_sents(predictions[cand_idx], nlp)))
        num_pred_sents = len(pred_sents)
        try:
            frags = [edus_no_space[i] for i in ei]
        except:
            print('EDU index error')
            continue
        sent_idx = []
        valid = True
        for frag in frags:
            found = False
            for sidx, sent in enumerate(source_sents_no_space):
                if frag in sent:
                    sent_idx.append(sidx)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            # num_valid += 1 =
            outs.append({
                'num_implied_edu': len(ei),
                'sent_pos': float(np.mean([x + 1 for x in sent_idx])),
                'num_implied_sent': len(set(sent_idx)),
                'num_pred_sents': num_pred_sents,
                'fusion_score': len(set(sent_idx)) / num_pred_sents,
                # 'fusion_score': len(set(sent_idx)) / len(ei),
            })
    return outs


if __name__ == '__main__':
    dtype = 'diverse'
    dataset = 'cnn_dailymail'
    if dtype == 'ea':
        experiment = 'cnn_e_v1'
        fn = 'test_from_beam_16_extract_cnn_ea_rand_v2'
    else:
        experiment = 'bart_large_cnn'
        fn = f'test_{dtype}_16_outputs'

    prefix = 'implied_'
    use_oracle = False

    path = os.path.join(f'/nlp/projects/faithsum/results/{experiment}/{fn}.csv')
    df = pd.read_csv(path)

    test = load_from_disk(f'/nlp/projects/faithsum/{dataset}_edu_alignments')['test']
    assert test['dataset_idx'] == list(range(len(test)))

    source_edu_annotated = test['source_edu_annotated']
    oracle_idxs = test['oracle_gain_idxs']

    full_beams = list(range(16))
    first = [0, 1, 2, 3]
    second = [4, 5, 6, 7,]
    third = [8, 9, 10, 11,]
    fourth = [12, 13, 14, 15]
    print(experiment)
    print(fn)
    for beam_list in [full_beams]:  #, first, second, third, fourth]:
        nlp = spacy.load('en_core_web_sm')
        num_invalid = 0
        num_valid = 0
        assert df['dataset_idx'].tolist() == list(range(len(test)))
        records = df.to_dict('records')
        avg_sent_pos = []
        if use_oracle:
            outputs = list(filter(None, list(p_uimap(
                lambda record: process_oracle(
                    record, nlp, source_edu_annotated,
                    oracle_idxs=oracle_idxs[record['dataset_idx']],
                ), records))))
        else:
            outputs = list(itertools.chain(*list(p_uimap(
                lambda record: process(
                    record, nlp, source_edu_annotated, prefix=prefix,
                    beams=beam_list
                ), records))))
        out_df = pd.DataFrame(outputs)
        print(beam_list)
        # print(out_df['sent_pos'].mean())
        print(out_df['num_pred_sents'].mean())
        print(out_df['num_implied_sent'].mean())
        # delta = (out_df['num_implied_sent'] - out_df['num_pred_sents']).mean()
        # print(delta)
        print(out_df['fusion_score'].mean())
        # print(out_df['fusion_score'].mean())
        print('\n')
