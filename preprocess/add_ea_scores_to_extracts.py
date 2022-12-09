import os
import ujson

import argparse
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
    parser.add_argument('--abstract_experiment', default='samsum_from_bert_red_extract_w_unlike')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--oracle_col', default='oracle_idxs_bert')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    args.convert_last_to_unprompted = False
    rouge = load_metric('rouge')
    nlp = spacy.load('en_core_web_sm')

    infer_dataset(args, 'abstract_experiment')
    infer_hf_model(args)

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

    additional_ids = tokenizer.additional_special_tokens_ids
    special_id_min = 999999 if len(additional_ids) == 0 else min(tokenizer.additional_special_tokens_ids)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
    ).to(args.device).eval()

    if 'pegasus' not in args.hf_model:
        model = model.half()

    for split in args.splits.split(','):
        data_split = dataset[split]
        dataset_idx2id = data_split['id']
        all_source_annotated = data_split['source_annotated']
        all_references = data_split[target_col]

        bert_suffix = '_bert' if 'bert' in args.oracle_col else ''
        in_fn = os.path.join(oracle_dir, f'{split}_candidates{bert_suffix}.json')
        print(f'Loading examples from {in_fn}')
        with open(in_fn, 'r') as fd:
            sampled_oracles = ujson.load(fd)

        for dataset_id, oracles in tqdm(sampled_oracles.items(), total=len(sampled_oracles)):
            dataset_idx = dataset_idx2id.index(dataset_id)
            source_annotated = all_source_annotated[dataset_idx]
            # Get source tokens
            reference = all_references[dataset_idx]
            extract_idx = [x['extract_idx'] for x in oracles]
            gen_outputs = gen_from_guide(
                args, nlp, model, tokenizer, source_annotated, reference,
                extract_idx, special_id_min, num_return_sequences=1
            )

            sampled_oracles[dataset_id] = {'ea': gen_outputs, 'oracles': oracles}
        print(f'Dumping examples with ea information to {in_fn}')
        with open(in_fn, 'w') as fd:
            ujson.dump(sampled_oracles, fd)
