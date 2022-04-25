import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import regex as re
import string

PUNCTUATION_TOKS = set(string.punctuation)

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
import itertools
import torch
from transformers import AutoTokenizer, BartTokenizer

from datasets import load_dataset
from data_utils import get_path_from_exp
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus
from sum_constants import summarization_name_mapping


def rouge_clean(s):
    return re.sub(r"[^a-zA-Z0-9 ]", "", s)


GEN_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        'num_beams': 4,
        'max_length': 142,
        'min_length': 56,
    },
}

SAMPLE_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        'max_length': 142,
        'min_length': 56,
        # 'top_p': 0.92,
        # 'top_k': 0,
        # 'do_sample': True,
        # https://github.com/andrejmiscic/simcls-pytorch/blob/2f0f8e00636f3eeebe2d41b4d318744a89602959/src/model.py
        'num_beam_groups': 16,
        'num_beams': 16,
        'diversity_penalty': 1.0,
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BART/PEGASUS Generator & Evaluator.')
    parser.add_argument('--wandb_name', required=True)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--per_device_eval_bs', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=1024)
    # Beam Search or Nucleus Sampling (more diverse)
    parser.add_argument('-sample_gen', default=False, action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_mask', default=5, type=int)
    parser.add_argument('--length_penalty', default=2.0, type=float)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('-use_hf_rouge', default=False, action='store_true')  # Much faster to use HF implementation
    parser.add_argument('--hf_model', default='Yale-LILY/brio-cnndm-uncased', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'Yale-LILY/brio-cnndm-uncased',
    ])
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # These are ignored but need to pass something in
    if args.experiment is None:
        args.experiment = args.wandb_name
    weight_dir = os.path.join(args.data_dir, 'weights')
    results_dir = os.path.join(args.data_dir, 'candidates', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device

    input_col, target_col = summarization_name_mapping[args.dataset]

    # Generating from this pre-trained model
    if args.wandb_name == 'brio' and args.hf_model == 'Yale-LILY/brio-cnndm-uncased':
        args.summary_style = 'abstract'
        args.lr = 1.0   # Needed to load
        tokenizer = BartTokenizer.from_pretrained(args.hf_model)
        model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model).to(gpu).eval()
    else:
        ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
        tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
        print(f'Loading tokenizer from {tokenizer_dir}...')
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)
        except:  # BRIO model doesn't load from AutoTokenizer
            tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

        print(f'Loading model from {ckpt_path}...')
        model = TransformerSummarizer.load_from_checkpoint(
            checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu).eval()

    # TODO why do we need this
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    split = dataset[args.split]

    batch_size = 1
    nlp = spacy.load('en_core_web_sm')
    n = len(split)
    sent_tok_ids = set(tokenizer.additional_special_tokens_ids)
    for batch_start in range(0, n, batch_size):
        end = min(batch_start + batch_size, n)
        batch_dataset_idxs = list(range(batch_start, end))
        batch_masks = []
        batch_input_ids = []
        for batch_idx in batch_dataset_idxs:
            example = split[batch_idx]
            inputs = example[input_col].strip()
            target = example[target_col]
            source_sents = [x for x in list(nlp(inputs).sents) if len(x) > 0]
            source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
            source_sents_tok_lower = [[token.lower() for token in sentence] for sentence in source_sents_tok]
            source_sents = list(map(str, source_sents))
            num_source = len(source_sents)

            target_sents = list(nlp(target).sents)
            target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
            target_toks_flat = list(itertools.chain(*target_sents_tok))
            target_toks_flat_lower = list(map(lambda x: x.lower(), target_toks_flat))
            target_vocab = set(target_toks_flat_lower) - PUNCTUATION_TOKS

            source_sent_tok_precision = []

            for source_sent_idx in range(num_source):
                sent_toks_lower = set(source_sents_tok_lower[source_sent_idx]) - PUNCTUATION_TOKS
                num_toks = len(sent_toks_lower)
                denom = max(1, num_toks)
                num_intersect = len(sent_toks_lower.intersection(target_vocab))
                source_sent_tok_precision.append(num_intersect / denom)

            source_sent_tok_precision = np.array(source_sent_tok_precision)
            k = min(num_source, args.num_mask)
            topk_source_indices = list((-source_sent_tok_precision).argsort()[:k])

            source_sents_space = [' ' + x if idx > 0 else x for idx, x in enumerate(source_sents)]
            source_sent_ids = list(map(tokenizer.encode, source_sents))
            bos, eos = source_sent_ids[0][0], source_sent_ids[0][-1]
            source_sent_ids = [x[1:-1] for x in source_sent_ids]  # Remove special tokens
            num_ids_per_sent = list(map(len, source_sent_ids))
            curr_offset = 1  # Start after BOS token
            sent_ranges = []
            for sent_len in num_ids_per_sent:
                sent_ranges.append((
                    curr_offset,
                    curr_offset + sent_len
                ))
                curr_offset += sent_len

            input_ids = [bos] + list(itertools.chain(*source_sent_ids)) + [eos]

            # Create the masks
            mask_idxs = np.sort(topk_source_indices)

            batch_input_ids.append(input_ids)
            num_ids = len(input_ids)
            batch_masks.append([sent_ranges[mask_idx] for mask_idx in mask_idxs])

        seq_lens = list(map(len, batch_input_ids))
        max_len = min(args.max_input_length, max(seq_lens))

        batch_input_ids_pad = np.zeros([batch_size, max_len], dtype=np.long)
        batch_input_ids_pad.fill(tokenizer.pad_token_id)
        for batch_idx, input_id_seq in enumerate(batch_input_ids):
            trunc_idx = min(max_len, len(input_id_seq))
            batch_input_ids_pad[batch_idx, :trunc_idx] = input_id_seq[:trunc_idx]
        batch_input_ids_pad = torch.from_numpy(batch_input_ids_pad).to(gpu)

        # encoder_inputs = {
        #     'input_ids': torch.from_numpy()
        #     'attention_mask': batch['attention_mask']
        # }
        # encoder_h = self.model.model.encoder(**encoder_inputs).last_hidden_state
        #
        # with torch.no_grad():
        #     encoder_states = model


        # encoder_attention_mask = np.ones(batch_size * args.num_cand, x)
        # Implement Padding

