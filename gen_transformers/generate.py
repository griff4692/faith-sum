import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
import spacy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from gen_transformers.dataset import SummaryDataModule
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus


GEN_KWARGS = {
    'cnn_dailymail': {
        # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
        # 'num_beams': 4,
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
        # 'top_p': 0.92,
        # 'top_k': 0,
        # 'do_sample': True,
    },
}


def get_path_from_exp(weights_dir, experiment, last=False):
    dir = os.path.join(weights_dir, experiment)
    paths = list(map(str, list(Path(dir).rglob('*.ckpt'))))
    if last:
        return [p for p in paths if 'last' in p][0]
    paths = [p for p in paths if 'last' not in p]
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')


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
    parser.add_argument('-add_sent_toks', default=False, action='store_true')

    parser = TransformerSummarizer.add_model_specific_args(parser)

    args = parser.parse_args()

    # TODO Support batches for predictions (simple fix)
    args.per_device_eval_batch_size = 1

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

    print(f'Loading tokenizer from {args.hf_model}...')
    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu).eval()
    datamodule = SummaryDataModule(args, tokenizer)
    model.on_predict_start()
    dataloader = datamodule.test_dataloader(max_examples=args.max_examples)
    outputs = []
    gen_kwargs = GEN_KWARGS[args.dataset]
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
        with torch.no_grad():
            batch_stats = model.predict_step(batch, **gen_kwargs)
        if type(batch_stats) == list:
            outputs += batch_stats
        else:
            outputs.append(batch_stats)

    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(results_dir, 'outputs.csv')
    print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
    num_col = outputs.select_dtypes('number')
    for col in list(num_col.columns):
        print(f'{col}: {num_col[col].dropna().mean()}')
