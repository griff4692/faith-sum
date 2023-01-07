import os
import regex as re

import argparse
import ujson
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer

from preprocess.extract_oracles import convert_to_sents
from sum_constants import summarization_name_mapping


def get_ids(
        args, split, tokenizer, batch_data, max_input_length=1024,
        max_output_length=256,
):

    source_edus = []
    target_edus = []
    source_annotated = []
    target_annotated = []
    for id in batch_data['id']:
        fn = os.path.join(args.data_dir, 'edu', split, f'{id}.json')
        with open(fn, 'r') as fd:
            edus = ujson.load(fd)

        sedu = [x for x in edus if x['dtype'] == 'source']
        tedu = [x for x in edus if x['dtype'] == 'target']

        sedu = list(sorted(sedu, key=lambda x: x['sent_idx']))
        tedu = list(sorted(tedu, key=lambda x: x['sent_idx']))

        tps = re.split(r'(<e>)', ' '.join(sedu))
        sedu_arr = [
            s.strip() for i, s in enumerate(tps) if i > 0 if tps[i - 1] == '<e>'
        ]

        source_edus.append(sedu_arr)
        target_edus.append(tedu)



    input_ids = tokenizer(
        source_edus,
        truncation=True,
        max_length=max_input_length,
    )['input_ids']
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    num_sents = [
        len(re.findall('<s\d+>', x)) for x in decoded
    ]
    labels = tokenizer(
        batch_data['target'],
        truncation=True,
        max_length=max_output_length,
    )['input_ids']

    row = {
        'source_annotated': source_annotated,
        'input_ids': input_ids,
        'labels': labels,
        'num_source_sents_pre_trunc': [len(x) for x in batch_source_sents],
        'num_source_sents': num_sents,
    }

    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='xsum')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default='facebook/bart-large-xsum', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'facebook/bart-large-cnn',
        'facebook/bart-large-xsum',
        'google/pegasus-xsum',
        'lidiya/bart-large-xsum-samsum',
    ])
    parser.add_argument('--num_proc', default=16, type=int)
    parser.add_argument('--max_num_sents', default=200, type=int)

    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    input_col, target_col = summarization_name_mapping[args.dataset]
    if 'pegasus' in args.hf_model:
        out_dir = os.path.join(args.data_dir, args.dataset + '_pegasus')
    else:
        out_dir = os.path.join(args.data_dir, args.dataset)
    print(f'Saving to {out_dir}')

    if args.dataset == 'xsum':
        max_input_length = 512
        max_output_length = 64
    else:
        max_input_length = 1024
        max_output_length = 256

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
            args, nlp, tokenizer, examples, input_col, target_col,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        ), batched=True, batch_size=1000, num_proc=args.num_proc)
        encoded = encoded.filter(lambda example: len(example[input_col].strip()) > 0)
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
