import os

import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from gen_transformers.dataset import get_sent_ngrams
from transformers import AutoTokenizer, BartTokenizer

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')
from data_utils import get_path_from_exp
from gen_transformers.model import TransformerSummarizer
from gen_transformers.gen_from_extract import get_extract_idxs_from_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking Scores for Inference.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='brio_just_rank')
    parser.add_argument('--rank_experiment', default='gen_extract_full_ar_mask_red_feat')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--split', default='validation')
    parser.add_argument('--hf_model', default='facebook/bart-base')
    parser.add_argument('--max_examples', default=999999, type=int)

    # Hyper-parameters: Should be the same as used when training
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_num_sents', type=int, default=200)

    args = parser.parse_args()
    weight_dir = os.path.join(args.data_dir, 'weights')

    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)
    except:  # BRIO model doesn't load from AutoTokenizer
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
    ).to(args.gpu_device).eval()

    results_dir = os.path.join(args.data_dir, 'results', args.rank_experiment)
    rank_fn = os.path.join(results_dir, f'{args.split}_sample_outputs.csv')
    print(f'Loading in predictions from {rank_fn}')
    outputs = pd.read_csv(os.path.join(results_dir, f'{args.split}_sample_outputs.csv'))
    outputs.dropna(subset=['extract'], inplace=True)
    n = len(outputs)
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=111)
        n = len(outputs)

    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)[args.split]
    dataset_idx2id = dataset['id']
    all_source_annotated = dataset['source_annotated']
    all_input_ids = dataset['input_ids']
    records = outputs.to_dict('records')
    special_id_min = min(tokenizer.additional_special_tokens_ids)

    for record in tqdm(records, total=len(records)):
        # sent_scores = np.array(list(map(float, record['sent_scores'].split(','))))
        source_annotated = all_source_annotated[record['dataset_idx']]
        source_ngrams = get_sent_ngrams(source_annotated)
        input_ids = all_input_ids[record['dataset_idx']]
        # Get source tokens
        reference = record['reference']

        # cls_mask = input_ids >
        extract_idx = [get_extract_idxs_from_str(x) for x in record['extract_idx'].split('<cand>')]

        encoder_inputs = {
            'input_ids': torch.LongTensor(input_ids).to(args.gpu_device).unsqueeze(0)
        }
        encoder_outputs = model.get_encoder_h(encoder_inputs)
        encoder_h = encoder_outputs.last_hidden_state
        cls_mask = encoder_inputs['input_ids'] >= special_id_min
        cls_h = encoder_h[cls_mask].unsqueeze(0)
        seq_len = cls_h.size()[1]

        extract_rouges = [float(x) for x in record['extract_rouges'].split(',')]

        num_cand = len(extract_idx)
        for i in range(num_cand):
            extract_idx[i].append(seq_len)

        max_len = max([len(x) for x in extract_idx])

        brio_labels = np.zeros([num_cand, max_len], dtype=np.int64)
        brio_labels.fill(-100)
        for cand_idx in range(num_cand):
            num = len(extract_idx[cand_idx])
            brio_labels[cand_idx, :num] = extract_idx[cand_idx]
        brio_labels = torch.from_numpy(brio_labels).to(args.gpu_device)

        stop_input_id = torch.LongTensor([0]).to(args.gpu_device)
        inputs_embeds = torch.cat([cls_h, model.stop_embed(stop_input_id).unsqueeze(0)], dim=1).repeat(num_cand, 1, 1)
        model.sent_bart.source_ngrams_cache = source_ngrams
        output = model.sent_bart(inputs_embeds=inputs_embeds, calculate_loss=False, labels=brio_labels)
