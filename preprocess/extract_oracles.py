import argparse
import os
from p_tqdm import p_uimap
import pandas as pd
import spacy
from tqdm import tqdm

from preprocess.convert_abstractive_to_extractive import gain_selection
from sum_constants import summarization_name_mapping
from datasets import load_dataset


def convert_to_sents(string, nlp):
    return [x for x in list(nlp(string.strip()).sents) if len(x.text.strip()) > 0]


def gen_oracle(args, example, nlp):
    input_col, target_col = summarization_name_mapping[args.dataset]
    inputs = example[input_col].strip()
    target = example[target_col].strip()
    source_sents = convert_to_sents(inputs, nlp)
    source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
    target_sents = convert_to_sents(target, nlp)
    target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
    # Sort oracle order or not
    oracle_idxs, oracle_rouge, r1_hist, r2_hist, best_hist = gain_selection(
        source_sents_tok, target_sents_tok, 5, lower=True, sort=False)
    output = {
        'id': example['id'],
        'sent_idxs': ','.join([str(x) for x in oracle_idxs]),
        'rouge1_history': r1_hist,
        'rouge2_history': r2_hist,
        'best_by_step': best_hist
    }
    output.update(oracle_rouge)  # ROUGE Scores
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

    out_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    print(f'Creating directory to store pre-computed oracles -> {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(','):
        data_split = dataset[split]
        if args.debug:
            data_split = data_split.select(list(range(128)))
        print(f'Processing {len(data_split)} {split} examples')
        if args.debug:
            outputs = pd.DataFrame(list(tqdm(map(
                lambda example: gen_oracle(args, example=example, nlp=nlp), data_split), total=len(data_split))))
        else:
            outputs = pd.DataFrame(list(p_uimap(
                lambda example: gen_oracle(args, example=example, nlp=nlp), data_split
            )))
        out_fn = os.path.join(out_dir, f'{split}_v2.csv')
        print(f'Saving {len(outputs)} examples to {out_fn}')
        outputs.to_csv(out_fn, index=False)
