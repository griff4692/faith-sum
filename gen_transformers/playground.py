import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import spacy

from gen_transformers.dataset import SummaryDataModule
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus
from constants import summarization_name_mapping
from gen_transformers.generate import get_path_from_exp
from convert_abstractive_to_extractive import gain_selection


GEN_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        'num_beams': 4,
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
    },
}

SAMPLE_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
        'top_p': 0.92,
        'top_k': 0,
        'do_sample': True,
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experiment with Models.')
    parser.add_argument('--wandb_name', default='cnn_bart_extract_prefix')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('-add_sent_toks', default=False, action='store_true')
    parser.add_argument('-sample_gen', default=False, action='store_true')

    parser = TransformerSummarizer.add_model_specific_args(parser)

    args = parser.parse_args()

    # TODO Support batches for predictions (simple fix)
    args.per_device_eval_batch_size = 1

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    if args.gpu_device is not None and args.gpu_device not in free_gpus:
        print(f'Warning! Youve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
        gpu = free_gpus[0]

    if args.experiment is None:
        args.experiment = args.wandb_name
    weight_dir = os.path.join(args.data_dir, 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)
    print(f'Loading tokenizer from {args.hf_model}...')
    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu).eval()

    # Dataset
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset('xsum')

    validation = dataset['validation']
    example = validation[0]
    input_col, target_col = summarization_name_mapping[args.dataset]

    source = example[input_col]
    target = example[target_col]
    nlp = spacy.load('en_core_web_sm')

    source_sents = list(nlp(source).sents)
    source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
    target_sents = list(nlp(target).sents)
    target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
    source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])

    # Sort oracle order or not
    oracle = gain_selection(source_sents_tok, target_sents_tok, 5, lower=True, sort=True)
    target_prefix = ''.join([f'<s{i}>' for i in oracle[0]]).strip()
    target_annotated = f'{target_prefix}<sep>{target}'

    inputs = tokenizer(source, truncation=True, max_length=1024, return_tensors='pt')

    input_ids = inputs['input_ids'].to(gpu)
    attention_mask = inputs['attention_mask'].to(gpu)
    # # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
    # 'num_beams': 4,
    # 'length_penalty': 4.0,
    # 'max_length': 142,
    # 'min_length': 56,
    with torch.no_grad():
        outputs = model.generate(

        )

