import argparse
import os

import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge Extracts and Abstracts')

    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_extract', default='gen_extract_full')
    parser.add_argument('--wandb_abstract', default='gen_abstract_full')

    args = parser.parse_args()

    extract_results_dir = os.path.join(args.data_dir, 'results', args.wandb_extract)
    abstract_results_dir = os.path.join(args.data_dir, 'results', args.wandb_abstract)
    extracts = pd.read_csv(os.path.join(extract_results_dir, 'validation_sample_outputs.csv'))
    abstracts = pd.read_csv(os.path.join(abstract_results_dir, 'validation_sample_outputs.csv'))
    beam_extracts = pd.read_csv(os.path.join(extract_results_dir, 'validation_beam_outputs.csv'))
    # beam_abstracts = pd.read_csv(os.path.join(abstract_results_dir, 'validation_beam_outputs.csv'))
    n = len(abstract_results_dir)

    extract_dataset_idxs = set(extracts['dataset_idx'].unique())
    abstract_dataset_idxs = set(extracts['dataset_idx'].unique())

    shared_idxs = extract_dataset_idxs.intersection(abstract_dataset_idxs)

    dataset_idxs = list(sorted(list(shared_idxs)))
    print(f'{len(extract_dataset_idxs)} Extracts.  {len(abstract_dataset_idxs)} Abstracts. {len(dataset_idxs)} shared.')
    stats = []
    for dataset_idx in tqdm(dataset_idxs):
        abstract_record = abstracts[abstracts['dataset_idx'] == dataset_idx].to_dict('records')[0]
        extract_record = extracts[extracts['dataset_idx'] == dataset_idx].to_dict('records')[0]
        beam_extract_record = beam_extracts[beam_extracts['dataset_idx'] == dataset_idx].to_dict('records')[0]
        # beam_abstract_record = beam_abstracts[beam_abstracts['dataset_idx'] == dataset_idx].to_dict('records')[0]

        best_abstract_r1 = abstract_record['best_abstract_rouge1_f1']
        best_implied_r1 = abstract_record['best_implied_rouge1_f1']
        best_extract_r1 = extract_record['best_extract_rouge1_f1']
        beam_extract_r1 = beam_extract_record['extract_rouge1_f1']
        # beam_abstract_r1 = beam_abstract_record['rouge1_f1']
        ensemble_r1 = max(best_extract_r1, best_abstract_r1, best_implied_r1, beam_extract_r1)

        if best_extract_r1 == ensemble_r1:
            best_method = 'sample_extract'
        elif best_implied_r1 == ensemble_r1:
            best_method = 'sample_implied'
        elif beam_extract_r1 == ensemble_r1:
            best_method = 'beam_extract'
        # elif beam_abstract_r1 == ensemble_r1:
        #     best_method = 'beam_abstract'
        else:
            best_method = 'sample_abstract'

        stats.append({
            'best_abstract_r1': best_abstract_r1,
            'best_extract_r1': best_extract_r1,
            'best_implied_r1': best_implied_r1,
            'beam_extract_r1': beam_extract_r1,
            # 'beam_abstract_r1': beam_abstract_r1,
            'ensemble_rouge1_f1': ensemble_r1,
            'best_method': best_method
        })

    stats = pd.DataFrame(stats)
    for col in stats.columns:
        if col == 'best_method':
            print(stats[col].value_counts())
        else:
            print(col, ' -> ', stats[col].mean())
