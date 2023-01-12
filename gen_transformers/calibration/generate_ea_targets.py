import os
import ujson

import argparse
import numpy as np
from datasets import load_metric
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import spacy

from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import infer_hf_model
from gen_transformers.data_utils import get_path_from_exp, infer_dataset
from gen_transformers.gen_from_extract import gen_from_guide
from sum_constants import summarization_name_mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default=None)
    parser.add_argument('--splits', default='validation,test,train')
    parser.add_argument('--abstract_experiment', default=None)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--max_targets', default=8, type=int)

    args = parser.parse_args()
    rouge = load_metric('rouge')
    nlp = spacy.load('en_core_web_sm')

    infer_dataset(args, 'abstract_experiment')
    infer_hf_model(args, is_abstract=True)

    print(f'Loading {args.dataset}...')
    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)
    _, target_col = summarization_name_mapping[args.dataset]

    oracle_dir = os.path.join(args.data_dir, args.dataset, 'oracle')

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
        print(f'Loading examples from {in_fn}')
        with open(in_fn, 'r') as fd:
            all_oracles = ujson.load(fd)

        for dataset_id, oracles in tqdm(all_oracles.items(), total=len(all_oracles)):
            dataset_idx = dataset_idx2id.index(dataset_id)
            source_annotated = all_source_annotated[dataset_idx]
            # Get source tokens
            reference = all_references[dataset_idx]

            oracle_sample = oracles
            if len(oracles) > args.max_targets:
                oracle_sample = list(np.random.choice(oracles, size=(args.max_targets), replace=False))

            extract_idx = [x['extract_idx'] for x in oracle_sample]
            gen_outputs = gen_from_guide(
                args, nlp, model, tokenizer, source_annotated, reference,
                extract_idx, num_return_sequences=1
            )

            all_oracles[dataset_id] = {'ea': gen_outputs, 'oracles': oracle_sample}
        print(f'Dumping examples with EA information to provide calibration targets to {in_fn}')
        print('Can now re-train Extract model with -add_brio_loss')
        with open(in_fn, 'w') as fd:
            ujson.dump(all_oracles, fd)