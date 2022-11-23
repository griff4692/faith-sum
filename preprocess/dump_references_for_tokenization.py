import os

import argparse
from tqdm import tqdm
from datasets import load_dataset


from sum_constants import summarization_name_mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenize Referencse')

    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--split', default='test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')

    args = parser.parse_args()
    _, target_col = summarization_name_mapping[args.dataset]
    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    out_fn = os.path.join(args.data_dir, args.dataset, f'{args.split}.target')

    references = dataset[args.split][target_col]

    references_clean = []
    for reference in tqdm(references):
        reference = reference.strip()
        if '\n' in reference:
            raise Exception('How to deal with this?')
        references_clean.append(reference)

    print(f'Writing references line by line to {out_fn}')
    with open(out_fn, 'w') as fd:
        fd.write('\n'.join(references_clean))
