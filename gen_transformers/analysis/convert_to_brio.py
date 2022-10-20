import os
import pandas as pd
import ujson
from tqdm import tqdm
import numpy as np

import argparse
from datasets import load_from_disk

import spacy


def brio_tokenize(text, nlp):
    doc = nlp(text)
    sents_untok = list(doc.sents)
    sents_untok = [str(x).strip() for x in sents_untok if len(str(x).strip()) > 0]
    sents_tok = [' '.join([str(y) for y in x]) for x in doc.sents]
    sents_tok = [x.strip() for x in sents_tok if len(x.strip()) > 0]
    return sents_untok, sents_tok


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return np.array([float(y) for y in num_str.split(delim)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert fn to BRIO')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum/results/')
    parser.add_argument('--extract_experiment', default='add_doc_bart_large_cnn')
    parser.add_argument('--fn', default='test_from_beam_16_extract.csv')
    parser.add_argument('--prediction_col', default='from_extract_abstract')

    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')

    fn = os.path.join(args.data_dir, args.extract_experiment, args.fn)
    df = pd.read_csv(fn)
    dataset = load_from_disk('/nlp/projects/faithsum/cnn_dailymail')['test']
    articles = dataset['article']
    # highlights = dataset['highlights']
    records = df.to_dict('records')

    out_dir = os.path.join(args.data_dir, args.extract_experiment, 'diverse', 'test')
    os.makedirs(out_dir, exist_ok=True)
    for idx, record in tqdm(enumerate(records), total=len(records)):
        dataset_idx = record['dataset_idx']
        candidates = record[args.prediction_col].split('<cand>')

        article_untok = articles[dataset_idx]
        reference_untok = record['reference']
        from_extract_rouges = get_arr(record['eval_from_extract_abstract_rouge1_f1'])

        article_untok, article_tok = brio_tokenize(article_untok, nlp)
        reference_untok, reference_tok = brio_tokenize(reference_untok, nlp)

        # Tokenize reference
        candidates_untok = []
        candidates_tok = []
        for cand_idx, cand_untok in enumerate(candidates):
            cand_untok, cand_tok = brio_tokenize(cand_untok, nlp)
            rouge = from_extract_rouges[cand_idx]
            candidates_untok.append([cand_untok, rouge])
            candidates_tok.append([cand_tok, rouge])

        obj = {
            'article': article_tok,
            'article_untok': article_untok,
            'abstract': reference_tok,
            'abstract_untok': reference_untok,
            'candidates': candidates_tok,
            'candidates_untok': candidates_untok,
        }

        out_fn = os.path.join(out_dir, f'{idx}.json')
        with open(out_fn, 'w') as fd:
            ujson.dump(obj, fd)
