import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
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
        'num_return_sequences': 1,
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
    },
}

SAMPLE_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        # 'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
        # 'top_p': 0.92,
        # 'top_k': 0,
        # 'do_sample': True,
        # 'num_return_sequences': 10  # Over-generate to get upper bound
        # https://github.com/andrejmiscic/simcls-pytorch/blob/2f0f8e00636f3eeebe2d41b4d318744a89602959/src/model.py
        'num_return_sequences': 16,
        'num_beam_groups': 16,
        'num_beams': 16,
        'diversity_penalty': 1.0,
        'length_penalty': 2.,
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BART/PEGASUS Generator & Evaluator.')
    parser.add_argument('--wandb_name', default='cnn_bart_extract_prefix')
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
    parser.add_argument('-add_sent_toks', default=False, action='store_true')
    parser.add_argument(
        '--oracle_cutoff', default=0.75, type=float,
        help='For summary_style=hybrid_control, summaries with ranking above this will be trained as extracts'
             '(to generate the oracle extractive summary).  Below, abstracts (to generate original reference). '
    )
    parser.add_argument(
        '--summary_style',
        default='plan_abstract',
        choices=[
            'plan_abstract',
            'abstract_plan',
            'extract',
            'plan',
            'abstract',
            'hybrid_control',
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
    args.add_sent_toks = args.add_sent_toks or args.summary_style in {'plan_abstract', 'plan', 'abstract_plan'}

    # TODO Support batches for predictions (simple fix)
    args.per_device_eval_bs = 1
    args.per_device_train_bs = 1

    if args.experiment is None:
        args.experiment = args.wandb_name
    weight_dir = os.path.join(args.data_dir, 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    if args.gpu_device is not None and args.gpu_device not in free_gpus:
        print(f'Warning! Youve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
        gpu = free_gpus[0]

    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)
    except:  # BRIO model doesn't load from AutoTokenizer
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu).eval()

    # TODO remove
    # args.hf_model = 'Yale-LILY/brio-cnndm-uncased'
    # args.lr = 0.1
    # tokenizer = BartTokenizer.from_pretrained(args.hf_model)
    # model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model).to(gpu).eval()

    # TODO why do we need this
    model.hparams.summary_style = args.summary_style
    args.oracle_filter = False
    args.contrast_modes = ''
    datamodule = SummaryDataModule(args, tokenizer)
    model.on_predict_start()
    dataloader, dataset_idxs = datamodule.get_split(args.split, max_examples=args.max_examples)
    outputs = []
    gen_kwargs = SAMPLE_KWARGS[args.dataset] if args.sample_gen else GEN_KWARGS[args.dataset]
    assert len(dataloader) == len(dataset_idxs)
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
        with torch.no_grad():
            batch_stats = model.predict_step(batch, **gen_kwargs)
            batch_stats['dataset_idx'] = dataset_idxs[batch_idx]
        if type(batch_stats) == list:
            outputs += batch_stats
        else:
            outputs.append(batch_stats)

    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(results_dir, f'{args.split}_outputs.csv')
    print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
    num_col = outputs.select_dtypes('number')
    for col in list(num_col.columns):
        print(f'{col}: {num_col[col].dropna().mean()}')

    table_cols = [
        'abstract_rouge_f1',
        'abstract_rouge2_f1',
        'abstract_rougeL_f1',
        'extract_rouge1_f1',
        'extract_rouge2_f1',
        'extract_rougeL_f1',
        'implied_extract_rouge1_f1',
        'implied_extract_rouge2_f1',
        'implied_extract_rougeL_f1',
    ]

    out_str = ''
    for col in table_cols:
        try:
            v = outputs[col].dropna().mean()
            out_str += f'{round(v, 3)},'
        except:  # If the column doesn't exist (i.e., we are generating for an abstractive model, extract_ won't exist)
            out_str += ','
    out_str.strip(',')
    print(','.join(table_cols))
    print(out_str)
