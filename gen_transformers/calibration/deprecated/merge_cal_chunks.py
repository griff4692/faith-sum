import os

import argparse
import pandas as pd


def collect_and_save_to(read_fns, out_fn):
    print(f'Collecting ', read_fns)
    out_df = pd.concat([pd.read_csv(fn) for fn in read_fns])
    print(f'Saving {len(out_df)} merged rows to {out_fn}')
    out_df.to_csv(out_fn, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate From Extract')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--abstract_experiment', default='extract_indicators')
    parser.add_argument('--num_candidates', default=16, type=int)
    parser.add_argument('--extract_experiment', default='add_doc')
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('-add_abstract_experiment', default=False, action='store_true')

    args = parser.parse_args()

    # This should always be true now
    args.add_abstract_experiment = True

    results_dir = os.path.join(args.data_dir, 'results', args.extract_experiment)
    decode_suffix = args.decode_method + '_' + str(args.num_candidates)
    chunk_suffixes = [f'_chunk_{i}' for i in range(args.num_chunks)]

    # Extract Merging
    extract_fns = [
        os.path.join(results_dir, f'{args.split}_{decode_suffix}_outputs{suffix}.csv') for suffix in chunk_suffixes
    ]
    extract_out_fn = os.path.join(results_dir, f'{args.split}_{decode_suffix}_outputs.csv')
    collect_and_save_to(extract_fns, extract_out_fn)

    # Extract-Abstract Merging
    top_k_str = '' if args.top_k is None else f'_{args.top_k}'
    chunk_suffix = '' if args.chunk is None else f'_chunk_{args.chunk}'
    prompt_suffix = '_w_unprompted' if args.convert_last_to_unprompted else ''
    ea_fns = [os.path.join(
        results_dir,
        f'{args.split}_from_{decode_suffix}_extract{top_k_str}_{args.abstract_experiment}{prompt_suffix}{suffix}.csv'
    ) for suffix in chunk_suffixes]
    ea_out_fn = os.path.join(
        results_dir,
        f'{args.split}_from_{decode_suffix}_extract{top_k_str}_{args.abstract_experiment}{prompt_suffix}.csv'
    )
    collect_and_save_to(ea_fns, ea_out_fn)
