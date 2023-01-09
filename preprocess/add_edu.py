import os
import regex as re

import argparse
import ujson
from datasets import load_from_disk
from transformers import AutoTokenizer
from gen_transformers.model_utils import infer_hf_model


EDU_SPECIAL_TOKENS = ['<e>', '</e>']


def add_edus_and_ids(args, split, tokenizer, batch_data, max_input_length=1024, max_output_length=256):
    target_edu_annotated = []
    source_edu_annotated = []

    num_source_edus_pre_trunc = []
    for id in batch_data['id']:
        fn = os.path.join(args.data_dir, 'edu', args.dataset, split, f'{id}.json')
        assert os.path.exists(fn)
        with open(fn, 'r') as fd:
            edus = ujson.load(fd)

        sedu = [x for x in edus if x['type'] == 'source']
        tedu = [x for x in edus if x['type'] == 'target']

        source_sents_w_edu = list(sorted(sedu, key=lambda x: x['sent_idx']))
        target_sents_w_edu = list(sorted(tedu, key=lambda x: x['sent_idx']))

        num_source_edus_pre_trunc.append(len(source_sents_w_edu))
        flat_source_sents_w_edu = ' '.join(list(map(lambda x: x['sent_w_edu'], source_sents_w_edu)))
        target_edu_annotated.append(' '.join(list(map(lambda x: x['sent_w_edu'], target_sents_w_edu))))
        source_edu_annotated.append(flat_source_sents_w_edu)

    input_ids = tokenizer(
        source_edu_annotated,
        truncation=True,
        max_length=max_input_length,
    )['input_ids']

    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

    num_source_edus_post_trunc = [
        len(re.findall('<e>', x)) for x in decoded
    ]

    labels = tokenizer(
        batch_data['target'],
        truncation=True,
        max_length=max_output_length,
    )['input_ids']

    row = {
        'source_edu_annotated': source_edu_annotated,
        'target_edu_annotated': target_edu_annotated,
        'input_ids': input_ids,
        'labels': labels,
        # Tokenizer truncates > 1,024 token sources. We just record the pre and post trunc \# of EDUs
        # We will only compute oracle alignments up to truncated
        'num_edus_pre_trunc': num_source_edus_pre_trunc,
        'num_edus_post_trunc': num_source_edus_post_trunc,
    }

    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add EDUs to the Sentencized Datasets')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--num_proc', default=1, type=int)

    args = parser.parse_args()

    infer_hf_model(args, is_abstract=False)

    out_dir = os.path.join(args.data_dir, args.dataset + '_edus')
    print(f'Saving to {out_dir}')

    if args.dataset == 'xsum':
        max_input_length = 512
        max_output_length = 64
    else:
        max_input_length = 1024
        max_output_length = 256

    print(f'Loading {args.dataset}...')
    sent_dir = os.path.join(args.data_dir, args.dataset + '_sentences')
    dataset = load_from_disk(sent_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    # Start and End of EDU spans
    special_tokens_dict = {'additional_special_tokens': EDU_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    encoded_data = {}
    for split in args.splits.split(','):
        # Filter dataset by which have been extracted
        filtered = dataset[split].filter(
            lambda ex: os.path.exists(os.path.join(args.data_dir, 'edu', args.dataset, split, ex['id'] + '.json')),
            batched=False, num_proc=args.num_proc,
        )

        print(f'Processing {len(filtered)}/{len(dataset[split])} {split} examples')
        encoded = filtered.map(
            lambda examples: add_edus_and_ids(
                args, split, tokenizer, examples, max_input_length=max_input_length,
                max_output_length=max_output_length,
            ),
            batched=True, batch_size=1000, num_proc=args.num_proc,
            remove_columns=['source_annotated', 'target_annotated'],
        )
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
