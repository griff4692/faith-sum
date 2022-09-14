import itertools
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from datasets import load_from_disk
import spacy
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import get_path_from_exp
from eval.rouge_metric import RougeMetric
from preprocess.convert_abstractive_to_extractive import gain_selection
from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import sentence_indicators
from preprocess.extract_oracles import convert_to_sents
from eval.diversity import diversity_score

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')
np.random.seed(1992)


def compute_rouge(generated, gold, rouge_metric, prefix=''):
    outputs = rouge_metric.evaluate_batch(generated, gold, aggregate=True)['rouge']
    f1s = []
    stats = {}
    for rouge_type in ['1', '2', 'L']:
        fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
        stats[f'{prefix}rouge{rouge_type}_precision'] = outputs[f'rouge_{rouge_type.lower()}_precision']
        stats[f'{prefix}rouge{rouge_type}_recall'] = outputs[f'rouge_{rouge_type.lower()}_recall']
        stats[f'{prefix}rouge{rouge_type}_f1'] = fscore
        f1s.append(fscore)
    stats[f'{prefix}mean_f1'] = np.array(f1s).mean()
    return stats


def get_idx(idx_str):
    idxs = idx_str.split(',')
    return list(map(int, idxs))


def gen_from_guide(model, tokenizer, source_annotated, idx_to_keep, special_id_min, num_return_sequences=1):
    inputs = tokenizer(
        [source_annotated] * len(idx_to_keep),
        padding='longest',
        truncation=True,
        max_length=1024,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(args.gpu_device)
    attention_mask = inputs['attention_mask'].to(args.gpu_device)
    cls_mask = input_ids >= special_id_min
    extract_indicators = []
    for cand_idx, extract_idx in enumerate(idx_to_keep):
        ei = sentence_indicators(cls_mask[cand_idx].unsqueeze(0), extract_idx, attention_mask[cand_idx].unsqueeze(0))
        extract_indicators.append(ei)
    extract_indicators = torch.cat(extract_indicators, dim=0)
    encoder_outputs = model.model.model.encoder(**{
        'input_ids': input_ids, 'attention_mask': attention_mask, 'extract_indicators': extract_indicators,
    })
    shared_kwargs = {
        'encoder_outputs': encoder_outputs,
        'attention_mask': attention_mask,
        'num_return_sequences': num_return_sequences,
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    }
    if num_return_sequences == 1:
        gen_kwargs = {
            'num_beams': 4,
        }
    else:
        gen_kwargs = {
            'diversity_penalty': 1.0,
            'num_beams': num_return_sequences,
            'num_beam_groups': num_return_sequences,
        }

    shared_kwargs.update(gen_kwargs)
    model = model.half()
    with torch.no_grad(), torch.cuda.amp.autocast():
        pred_ids = model.model.generate(**shared_kwargs)
    pred_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)

    cand_metrics, best_metric, avg_r1_f1, diversity = model.score_candidates(
        [reference], pred_str, prefix='from_extract', eval=True,
    )
    rouge1_f1s = [y['best_from_extract_rouge1_f1'] for y in cand_metrics]
    pred_rank = int(np.argmax(rouge1_f1s))
    diversity = diversity_score(pred_str)
    return {
        'best_from_extract_rouge1_f1': best_metric['best_from_extract_rouge1_f1'],
        'best_from_extract_rouge2_f1': best_metric['best_from_extract_rouge2_f1'],
        'best_from_extract_rougeL_f1': best_metric['best_from_extract_rougeL_f1'],
        'min_from_extract_rouge1_f1': min(rouge1_f1s),
        'avg_from_extract_rouge1_f1': float(np.mean(rouge1_f1s)),
        'from_extract_abstract': '<cand>'.join(pred_str),
        'from_extract_rouges': '<cand>'.join(map(str, rouge1_f1s)),
        'pred_rank': pred_rank,
        'diversity': diversity,
    }


def get_extract_idxs_from_str(extract_str):
    if len(extract_str) == 0:
        print('Empty Prediction. Predicting first sentence.')
        return [0]
    try:
        return list(map(int, extract_str.split(',')))
    except:
        print('Parse Error. Predicting first sentence.')
        return [0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate From Extract')

    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='extract_indicators')
    parser.add_argument('--extract_experiment', default='gen_extract_full_ar_mask_red_feat')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-do_not_save', default=False, action='store_true')
    parser.add_argument('--hf_model', default='facebook/bart-base')
    parser.add_argument('--max_examples', default=999999, type=int)
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--extract_mode', default='sample', choices=['sample', 'beam'])
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()

    results_dir = os.path.join(args.data_dir, 'results', args.extract_experiment)
    outputs = pd.read_csv(os.path.join(results_dir, f'{args.split}_{args.extract_mode}_outputs.csv'))
    outputs.dropna(subset=['extract'], inplace=True)
    n = len(outputs)
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=111)
        n = len(outputs)

    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)[args.split]
    dataset_idx2id = dataset['id']
    all_source_annotated = dataset['source_annotated']

    records = outputs.to_dict('records')
    weight_dir = os.path.join(args.data_dir, 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    additional_ids = tokenizer.additional_special_tokens_ids
    special_id_min = 999999 if len(additional_ids) == 0 else min(tokenizer.additional_special_tokens_ids)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(args.gpu_device).eval()

    rouge_metric = RougeMetric()
    df = []
    updated_records = []
    stats = []
    wins, losses = 0, 0
    compare_col = 'best_extract_rouge1_f1' if args.extract_mode == 'sample' else 'extract_rouge1_f1'
    for record in tqdm(records, total=len(records)):
        # sent_scores = np.array(list(map(float, record['sent_scores'].split(','))))
        source_annotated = all_source_annotated[record['dataset_idx']]
        # Get source tokens
        reference = record['reference']

        extract_idx = [get_extract_idxs_from_str(x) for x in record['extract_idx'].split('<cand>')]
        if args.top_k is not None and args.top_k < len(extract_idx):
            extract_idx = extract_idx[:args.top_k]
        gen_output = gen_from_guide(
            model, tokenizer, source_annotated, extract_idx, special_id_min,
            num_return_sequences=args.num_return_sequences
        )

        best_ensemble_rouge1_f1 = max(record[compare_col], gen_output['best_from_extract_rouge1_f1'])
        stat_row = {
            compare_col: record[compare_col],
            'best_from_extract_rouge1_f1': gen_output['best_from_extract_rouge1_f1'],
            'best_ensemble_rouge1_f1': best_ensemble_rouge1_f1,
        }

        if args.extract_mode == 'sample':
            more = {
                'avg_extract_rouge1_f1': record['avg_extract_rouge1_f1'],
                'avg_from_extract_rouge1_f1': gen_output['avg_from_extract_rouge1_f1'],
                'diversity': gen_output['diversity']
            }
            stat_row.update(more)
        stats.append(stat_row)

        if gen_output['best_from_extract_rouge1_f1'] >= record[compare_col]:
            wins += 1
        else:
            losses += 1
        gen_output['best_ensemble_rouge1_f1'] = best_ensemble_rouge1_f1
        record.update(gen_output)
        updated_records.append(record)
        print(f'Abstract Wins: {wins}. Losses: {losses}.')
    stats = pd.DataFrame(stats)
    avgs = {k: stats[k].mean() for k in stats.columns}
    print(avgs)
    winshare = avgs['best_from_extract_rouge1_f1'] - avgs[compare_col]
    print(f'Win share: {winshare}')

    updated_df = pd.DataFrame(updated_records)
    if not args.do_not_save:
        top_k_str = '' if args.top_k is None else f'_{args.top_k}'
        updated_out_fn = os.path.join(results_dir, f'{args.split}_from_{args.extract_mode}_extract.csv')
        print(f'Saving prompted abstracts to {updated_out_fn}')
        updated_df.to_csv(updated_out_fn, index=False)

    extract_tok_len = updated_df['extract'].apply(lambda x: len(
        x.split('<cand>')[0].split(' '))).mean()
    ref_tok_len = updated_df['reference'].apply(
        lambda x: len(x.split(' '))).mean()
    from_extract_tok_len = updated_df['from_extract_abstract'].apply(
        lambda x: len(x.split('<cand>')[0].split(' '))).mean()

    print(f'Average Extract Tokens: {extract_tok_len}')
    print(f'Average From Extract Abstract Tokens: {from_extract_tok_len}')
    print(f'Average Reference Tokens: {ref_tok_len}')
