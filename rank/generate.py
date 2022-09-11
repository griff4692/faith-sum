import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')
from tqdm import tqdm
import argparse
import torch
import pandas as pd
import numpy as np

from gen_transformers.data_utils import get_path_from_exp
from global_utils import get_free_gpus
from rank.dataset import RankDataModule
from rank.model import SummaryRanker
from rank.main import build_tokenizer


def run_inference(args):
    if args.gpu_device is not None:
        gpus = [args.gpu_device]
    else:
        gpus = get_free_gpus() if torch.cuda.is_available() and not args.cpu else None
        assert gpus is None or len(gpus) > 0
        if gpus is not None and (args.debug or args.find_lr):
            gpus = [gpus[0]]
        if gpus is not None and len(gpus) > args.max_gpus:
            gpus = gpus[:args.max_gpus]
        if gpus is not None:
            gpu_str = ','.join([str(x) for x in gpus])
            print(f'Using GPUS --> {gpu_str}...')

    gpu = gpus[0]
    experiment_dir = os.path.join(args.weight_dir, args.wandb_name, 'rank')
    ckpt_path = get_path_from_exp(experiment_dir)
    tokenizer = build_tokenizer(max_num_sents=args.max_num_sents)

    model = SummaryRanker.load_from_checkpoint(
        checkpoint_path=ckpt_path, args=args, tokenizer=tokenizer, strict=True
    ).to(gpu).eval()
    datamodule = RankDataModule(args, tokenizer=tokenizer)
    dataloader, _ = datamodule.get_split(args.split)

    split_fn = os.path.join(
        args.data_dir, 'results', args.gen_experiment, f'{args.split}_sample_outputs.csv'
    )
    split_predictions = pd.read_csv(split_fn)
    dataset_idx_to_record = {}
    for record in split_predictions.to_dict('records'):
        dataset_idx_to_record[record['dataset_idx']] = record

    outputs = []
    # No reason to over-generate
    num_pred = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch['inputs'] = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch['inputs'].items()}
        with torch.no_grad():
            dataset_idxs, pred_dist = model.predict_step(batch)
            for dataset_idx, dist in zip(dataset_idxs, pred_dist):
                prediction_record = dataset_idx_to_record[dataset_idx]
                dist = dist.tolist()
                pred_rank = [int(x) for x in np.argsort(-np.array(dist))]
                num_pred = max(num_pred, len(pred_rank))
                scores = [float(x) for x in prediction_record['extract_rouges'].split(',')]
                dist_str = ','.join([str(x) for x in dist])
                prediction_record['rank_pred'] = dist_str
                for i in range(1, len(pred_rank) + 1):
                    pred_scores = [scores[pred_idx] for pred_idx in pred_rank[:i]]
                    prediction_record[f'min_score_at_{i}'] = min(pred_scores)
                    prediction_record[f'mean_score_at_{i}'] = sum(pred_scores) / i
                    prediction_record[f'max_score_at_{i}'] = max(pred_scores)
                outputs.append(prediction_record)
    outputs = pd.DataFrame(outputs)
    out_fn = os.path.join(
        args.data_dir, 'results', args.gen_experiment,
        f'{args.split}_sample_outputs_with_{args.wandb_name}_ranking.csv'
    )
    print(f'Saving {len(outputs)} outputs to {out_fn}')
    outputs.to_csv(out_fn, index=False)

    for i in range(1, num_pred + 1):
        min_s = outputs[f'min_score_at_{i}'].dropna().mean()
        mean_s = outputs[f'mean_score_at_{i}'].dropna().mean()
        max_s = outputs[f'max_score_at_{i}'].dropna().mean()
        print(f'Rank={i}: min={min_s}, mean={mean_s}, max={max_s}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking Scores for Inference.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='gen_extract_sample_feats')
    parser.add_argument('--gen_experiment', default='gen_extract_full_ar_mask_red_feat')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--split', default='validation')
    parser.add_argument('--batch_size', default=4, type=int)

    # Hyper-parameters: Should be the same as used when training
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_num_sents', type=int, default=200)

    args = parser.parse_args()
    args.weight_dir = os.path.join(args.data_dir, 'weights')
    run_inference(args)
