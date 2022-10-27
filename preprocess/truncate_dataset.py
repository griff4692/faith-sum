import os
import regex as re

import argparse
import spacy

from sum_constants import summarization_name_mapping
from datasets import load_dataset
from transformers import AutoTokenizer
from preprocess.extract_oracles import convert_to_sents
from preprocess.convert_abstractive_to_extractive import gain_selection


def get_ids(args, nlp, tokenizer, batch_data, input_col, target_col, max_input_length=1024, max_output_length=256):
    batch_source_sents = [
        convert_to_sents(inputs, nlp, is_dialogue=args.dataset == 'samsum') for inputs in batch_data[input_col]
    ]
    source_annotated = [
        ''.join(
            [f'<s{i}> {s}' for i, s in enumerate(source_sents) if i < args.max_num_sents]
        ) for source_sents in batch_source_sents
    ]

    input_ids = tokenizer(
        source_annotated,
        truncation=True,
        max_length=max_input_length,
    )['input_ids']
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    num_sents = [
        len(re.findall('<s\d+>', x)) for x in decoded
    ]
    labels = tokenizer(
        batch_data[target_col],
        truncation=True,
        max_length=max_output_length,
    )['input_ids']

    oracle_idxs = []
    batch_size = len(source_annotated)
    oracle_rouge1 = []
    oracle_rouge2 = []
    rouge1_history = []
    rouge2_history = []
    best_history = []
    for batch_idx in range(batch_size):
        target = batch_data[target_col][batch_idx]
        target_sents = convert_to_sents(target, nlp, is_dialogue=args.dataset == 'samsum')
        source_sents = [sent for i, sent in enumerate(batch_source_sents[batch_idx]) if i < num_sents[batch_idx]]
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
        # Sort oracle order or not
        idxs, rouge, r1_hist, r2_hist, best_hist = gain_selection(
            source_sents_tok, target_sents_tok, 5, lower=True, sort=False)

        rouge1_history.append(r1_hist)
        rouge2_history.append(r2_hist)
        best_history.append(best_hist)
        oracle_idxs.append(idxs)
        oracle_rouge1.append(rouge['rouge_1'])
        oracle_rouge2.append(rouge['rouge_2'])

    return {
        'source_annotated': source_annotated,
        'input_ids': input_ids,
        'labels': labels,
        'num_source_sents_pre_trunc': [len(x) for x in batch_source_sents],
        'num_source_sents': num_sents,
        'oracle_idxs': oracle_idxs,
        'oracle_rouge1': oracle_rouge1,
        'oracle_rouge2': oracle_rouge2,
        'rouge1_history': rouge1_history,
        'rouge2_history': rouge2_history,
        'best_history': best_history,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'google/pegasus-large',
        'lidiya/bart-large-xsum-samsum',
    ])
    parser.add_argument('--max_num_sents', default=200, type=int)

    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    input_col, target_col = summarization_name_mapping[args.dataset]
    out_dir = os.path.join(args.data_dir, args.dataset)

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    add_tokens = [f'<s{i}>' for i in range(args.max_num_sents)]
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    encoded_data = {}
    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        encoded = dataset[split].map(lambda examples: get_ids(
            args, nlp, tokenizer, examples, input_col, target_col
        ), batched=True, num_proc=32)
        encoded = encoded.filter(lambda example: len(example[input_col].strip()) > 0)
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
