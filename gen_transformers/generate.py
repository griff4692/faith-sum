import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BartTokenizer

from data_utils import get_path_from_exp
from gen_transformers.dataset import SummaryDataModule
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus


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
    parser.add_argument('-do_not_save', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--per_device_eval_bs', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=1024)
    # Beam Search or Nucleus Sampling (more diverse)
    parser.add_argument('-sample_gen', default=False, action='store_true')
    parser.add_argument('-add_sent_toks', default=False, action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--length_penalty', default=2.0, type=float)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--num_return_sequences', default=16, type=int)
    parser.add_argument('--max_num_sents', type=int, default=200)
    parser.add_argument('--extract_method', type=str, default='generate', choices=['generate', 'select'])
    parser.add_argument('-use_hf_rouge', default=False, action='store_true')  # Much faster to use HF implementation
    parser.add_argument('--bootstraps', default=1, type=int)
    parser.add_argument('-add_sent_brio', default=False, action='store_true')
    parser.add_argument(
        '--summary_style',
        default='extract_abstract',
        choices=[
            'extract_abstract',
            'abstract',
            'extract'
        ], help='Target output during training. plan is a sequence of <s{idx}> tokens, extract is oracle summary, '
                'abstract is original reference'
    )
    parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'Yale-LILY/brio-cnndm-uncased',
    ])
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()
    args.add_sent_toks = args.add_sent_toks or 'extract' in args.summary_style

    np.random.seed(args.seed)

    # These are ignored but need to pass something in
    args.per_device_eval_bs = -1
    args.per_device_train_bs = -1

    if args.experiment is None:
        args.experiment = args.wandb_name
    weight_dir = os.path.join(args.data_dir, 'weights')
    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device

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
    model.hparams.summary_style = args.summary_style
    datamodule = SummaryDataModule(args, tokenizer)
    model.on_predict_start()

    exp_results = []
    for exp_id in range(args.bootstraps):
        # Override behavior during training
        dataloader_kwargs = {'shuffle': False, 'batch_size': args.batch_size}
        dataloader, dataset_idxs = datamodule.get_split(args.split, max_examples=args.max_examples, **dataloader_kwargs)
        outputs = []
        gen_kwargs = SAMPLE_KWARGS[args.dataset] if args.sample_gen else GEN_KWARGS[args.dataset]
        gen_kwargs['length_penalty'] = args.length_penalty
        gen_kwargs['use_hf_rouge'] = args.use_hf_rouge

        # No reason to over-generate
        gen_kwargs['num_return_sequences'] = args.num_return_sequences if args.sample_gen else 1
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
            start = args.batch_size * batch_idx
            actual_batch_size = len(batch['input_ids'])
            end = start + actual_batch_size
            batch_dataset_idxs = dataset_idxs[start:end]
            with torch.no_grad():
                batch_stats = model.predict_step(batch, **gen_kwargs)
                for j in range(actual_batch_size):
                    batch_stats[j]['dataset_idx'] = batch_dataset_idxs[j]
                outputs += batch_stats

        outputs = pd.DataFrame(outputs)
        method = '_sample' if args.sample_gen else '_beam'
        out_fn = os.path.join(results_dir, f'{args.split}{method}_outputs.csv')
        if not args.do_not_save:
            print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
            outputs.to_csv(out_fn, index=False)
        num_col = outputs.select_dtypes('number')
        for col in list(num_col.columns):
            print(f'{col}: {num_col[col].dropna().mean()}')

        table_cols = [
            'rouge1_f1',
            'rouge2_f1',
            'rougeL_f1',
            'extract_rouge1_f1',
            'extract_rouge2_f1',
            'extract_rougeL_f1',
            'implied_rouge1_f1',
            'implied_rouge2_f1',
            'implied_rougeL_f1',
        ]

        out_str = ''
        for col in table_cols:
            try:
                v = outputs[col].dropna().mean()
                out_str += f'{round(v, 3)},'
            except:  # If the column doesn't exist (i.e., we are generating for an abstractive model, extract_ won't exist)
                out_str += ','
        out_str.strip(',')
        # print(','.join(table_cols))
        # print(out_str)

        agg_cols = [
            'rouge1_f1', 'implied_rouge1_f1', 'extract_rouge1_f1',
            'best_extract_rouge1_f1', 'best_abstract_rouge1_f1', 'best_implied_rouge1_f1',
            'oracle_prompt_rouge1_f1', 'extract_implied_sent_f1', 'extract_gen_rouge1_f1',
            'rand_plan_implied_sent_f1', 'avg_rouge1_f1', 'avg_implied_rouge1_f1', 'avg_extract_rouge1_f1',
            'diversity', 'implied_diversity', 'extract_diversity'
        ]
        exp_row = {
            col: outputs[col].dropna().mean() for col in agg_cols if col in list(outputs.columns)
        }

        exp_results.append(exp_row)
    exp_results = pd.DataFrame(exp_results)
    out_fn = 'confidence_sample.csv' if args.sample_gen else 'confidence.csv'
    out_fn = args.split + '_' + out_fn
    print(args.length_penalty)
    if not args.do_not_save:
        exp_results.to_csv(out_fn, index=False)
    for col in list(exp_results.columns):
        print(f'{col}: min={exp_results[col].min()}, max={exp_results[col].max()}, avg={exp_results[col].mean()}')
