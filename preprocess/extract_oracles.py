import argparse
import os
from p_tqdm import p_uimap
import pandas as pd

from convert_abstractive_to_extractive import gain_selection
from constants import summarization_name_mapping
from datasets import load_dataset
import spacy


def gen_oracle(args, example, nlp):
    input_col, target_col = summarization_name_mapping[args.dataset]
    inputs = example[input_col]
    target = example[target_col]
    source_sents = list(nlp(inputs).sents)
    source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
    target_sents = list(nlp(target).sents)
    target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
    # Sort oracle order or not
    oracle = gain_selection(source_sents_tok, target_sents_tok, 5, lower=True, sort=True)
    output = {
        'id': example['id'],
        'sent_idxs': ','.join([str(x) for x in oracle[0]]),
    }
    output.update(oracle[1])  # ROUGE Scores
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')

    args = parser.parse_args()

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

    out_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(','):
        data_split = dataset[split]
        print(f'Processing {len(data_split)} {split} examples')
        outputs = pd.DataFrame(list(p_uimap(lambda example: gen_oracle(args, example=example, nlp=nlp), data_split)))
        out_fn = os.path.join(out_dir, f'{split}.csv')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs.to_csv(out_fn, index=False)
