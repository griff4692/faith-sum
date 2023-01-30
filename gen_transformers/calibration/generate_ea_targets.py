import os
import ujson

import argparse
import numpy as np
from datasets import load_metric
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import spacy

from gen_transformers.gen_from_extract import filter_out_extract_tags
from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import infer_hf_model
from gen_transformers.data_utils import get_path_from_exp, infer_dataset
from gen_transformers.gen_from_extract import gen_from_guide
from sum_constants import summarization_name_mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract EA scores for calibration')

    parser.add_argument('--dataset', default=None)
    parser.add_argument('--splits', default='validation,test,train')
    parser.add_argument('--abstract_experiment', default=None)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--max_targets', default=8, type=int)
    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=8, type=int)

    args = parser.parse_args()
    rouge = load_metric('rouge')
    nlp = spacy.load('en_core_web_sm')

    infer_dataset(args, 'abstract_experiment')
    infer_hf_model(args, is_abstract=True)

    chunk_suffix = '' if args.chunk is None else f'_chunk_{args.chunk}'

    print(f'Loading {args.dataset}...')
    data_dir = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    dataset = load_from_disk(data_dir)
    _, target_col = summarization_name_mapping[args.dataset]

    oracle_dir = os.path.join(args.data_dir, args.dataset, 'oracle')
    os.makedirs(oracle_dir, exist_ok=True)

    weight_dir = os.path.join(args.data_dir, 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.abstract_experiment)
    tokenizer_dir = os.path.join(weight_dir, args.abstract_experiment, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
    ).to(args.device).eval()

    if 'pegasus' not in args.hf_model:
        model = model.half()

    for split in args.splits.split(','):
        data_split = dataset[split]
        dataset_idx2id = data_split['id']
        all_source_annotated = data_split['source_edu_annotated']
        all_references = data_split[target_col]

        in_fn = os.path.join(oracle_dir, f'{split}_candidates.json')
        out_fn = os.path.join(oracle_dir, f'{split}_candidates_targets{chunk_suffix}.json')
        print(f'Loading examples from {in_fn}')
        with open(in_fn, 'r') as fd:
            all_oracles = ujson.load(fd)

        dataset_ids = list(sorted(list(all_oracles.keys())))
        if args.chunk is not None:
            dataset_ids = np.array_split(dataset_ids, args.num_chunks)[args.chunk]

        out_dict = {}
        for dataset_id in tqdm(dataset_ids):
            oracles = all_oracles[dataset_id]
            dataset_idx = dataset_idx2id.index(dataset_id)
            source_annotated = all_source_annotated[dataset_idx]
            # Get source tokens
            reference = all_references[dataset_idx]

            # If empty source, continue
            sa_clean = filter_out_extract_tags(source_annotated, [])
            if len(sa_clean.split(' ')) <= 5:
                print(dataset_id)
                print(f'Source is too short: {sa_clean}')
                continue

            if len(reference.split(' ')) <= 3:
                print(dataset_id)
                print(f'Reference is too short: {reference}')
                continue

            oracle_sample = oracles
            if len(oracles) > args.max_targets:
                oracle_sample = list(np.random.choice(oracles, size=(args.max_targets), replace=False))

            extract_idx = [x['idxs'] for x in oracle_sample]
            try:
                gen_outputs = gen_from_guide(
                    args, nlp, model, tokenizer, source_annotated, reference,
                    extract_idx, num_return_sequences=1
                )
            except Exception as e:
                print(f'Dataset ID: {dataset_id}')
                print(f'Source: {source_annotated}')
                print(f'Reference: {reference}')
                raise Exception(f'Gen from Guide Failed: {e}. See above inputs')

            out_dict[dataset_id] = {'ea': gen_outputs, 'oracles': oracle_sample}
        print(f'Dumping examples with EA information to provide calibration targets to {out_fn}')
        print('Can now re-train Extract model with -add_brio_loss')
        with open(out_fn, 'w') as fd:
            ujson.dump(out_dict, fd)
