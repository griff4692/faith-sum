import itertools
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import ujson

import pandas as pd
import argparse
import spacy
import numpy as np
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from scipy.special import expit

from data_utils import get_path_from_exp
from eval.rouge_metric import RougeMetric
from preprocess.convert_abstractive_to_extractive import gain_selection
from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import sentence_mask
from preprocess.extract_oracles import convert_to_sents
from datasets import load_dataset

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


def get_alignments(source_toks, summary, nlp):
    sum_sents = convert_to_sents(summary, nlp)
    sum_sents_tok = [[str(token.text) for token in sentence] for sentence in sum_sents]
    aligned_sents = list(map(lambda x: gain_selection(source_toks, [x], 3, lower=False, sort=False)[0], sum_sents_tok))
    aligned_sents_flat = list(set(list(itertools.chain(*aligned_sents))))
    return aligned_sents, aligned_sents_flat


def get_idx(idx_str):
    idxs = idx_str.split(',')
    return list(map(int, idxs))


def get_priority(source_toks, summary, nlp):
    sum_sents = convert_to_sents(summary, nlp)
    sum_sents_tok = [[str(token.text) for token in sentence] for sentence in sum_sents]
    gs = gain_selection(source_toks, sum_sents_tok, summary_size=0)
    sent_r1s = list(map(lambda x: float(x), gs[2].split(',')))
    sent_r2s = list(map(lambda x: float(x), gs[3].split(',')))
    assert len(sent_r1s) == len(source_toks)
    avg_rs = np.array([(a + b) / 2.0 for (a, b) in zip(sent_r1s, sent_r2s)])
    return np.argsort(- avg_rs), avg_rs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mask Cross Attention and Generate')

    parser.add_argument('--extractor', default='extract', choices=['oracle', 'extract'])
    parser.add_argument('--gpu_device', default=1, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='score_abstract_kld')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--hf_model', default='facebook/bart-base')
    parser.add_argument('--max_examples', default=1000, type=int)
    parser.add_argument('--k', default=10, type=int)

    args = parser.parse_args()

    oracle_df = pd.read_csv(os.path.join(args.data_dir, 'cnn_dailymail/oracle/validation_v2.csv'))
    results_dir = os.path.join(args.data_dir, 'results', args.wandb_name)
    outputs = pd.read_csv(os.path.join(results_dir, 'validation_beam_outputs.csv'))
    n = len(outputs)
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=1992)
    nlp = spacy.load('en_core_web_sm')

    dataset = load_dataset('cnn_dailymail', '3.0.0')['validation']
    dataset_idx2id = dataset['id']
    orig_sources = dataset['article']

    ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}
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
    n = len(records)
    sample_out = []
    rank_dataset = []
    for record in tqdm(records, total=n):
        oracle_obj = ids2oracles[dataset_idx2id[record['dataset_idx']]]
        source = orig_sources[record['dataset_idx']]
        source_sents = convert_to_sents(source, nlp)
        n = len(source_sents)

        sent_scores = np.array(list(map(float, record['sent_scores'].split(','))))
        num_trunc_sent = len(sent_scores)  # Up to 1024 tokens usually sometimes reduces number of sentences
        assert num_trunc_sent <= len(source_sents)
        source_sents = source_sents[:num_trunc_sent]

        rank_example = {
            'dataset_idx': record['dataset_idx'],
            'source_sents': [str(x) for x in source_sents],
            'reference': record['reference'],
            'abstract': record['abstract'],
            'priority': {
                'prob': [],
                'source_idxs': []
            },
            'sampled': {
                'source_idxs': [],
                'scores': [],
                'extracts': [],
                'abstracts': [],
            }
        }

        source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Get source tokens
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        pred_abstract = record['abstract']
        reference = record['reference']

        extract_priority = (-sent_scores).argsort()
        extract_scores = sent_scores[extract_priority]
        extract_scores_norm = expit(extract_scores)
        # extract_scores_norm[extract_scores_norm < 0.1] = 0.0
        # emin, emax = extract_scores_norm.min(), extract_scores_norm.max()
        # extract_scores_norm = (extract_scores_norm - emin) / (emax - emin)
        oracle_priority, oracle_scores = get_priority(source_sents_tok, reference, nlp)

        if args.extractor == 'extract':
            priority = extract_priority
        elif args.extractor == 'oracle':
            priority = oracle_priority

        inputs = tokenizer(
            [source_annotated],
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].to(args.gpu_device)
        attention_mask = inputs['attention_mask'].to(args.gpu_device)
        cls_mask = input_ids >= special_id_min

        idxs = []
        masks = []
        for exp in range(args.k):
            unmask_idxs = []
            for source_idx in range(len(extract_scores_norm)):
                should_select = np.random.random() <= extract_scores_norm[source_idx]
                if should_select:
                    unmask_idxs.append(extract_priority[source_idx])
            idxs.append(unmask_idxs)
            masks.append(sentence_mask(cls_mask, unmask_idxs))

        sampled_extracts = [''.join('<s{i}>' + str(source_sents[i]) for i in idx) for idx in idxs]
        all_masks = torch.cat(masks)
        num_cand = len(all_masks)
        input_ids_rep = input_ids.repeat(num_cand, 1)
        kwargs = {
            'input_ids': input_ids_rep,
            'attention_mask': all_masks,
            'num_return_sequences': 1,
            'num_beams': 4,
            'length_penalty': 4.0,
            'max_length': 142,
            'min_length': 56,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }

        pred_ids = model.model.generate(**kwargs)
        pred_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)

        rank_example['priority']['prob'] = list(extract_scores_norm)
        rank_example['priority']['source_idxs'] = list(extract_priority)
        rank_example['sampled']['source_idxs'] = idxs
        rank_example['sampled']['abstracts'] = pred_str
        rank_example['sampled']['extracts'] = sampled_extracts

        rouges = []
        r1s = []
        for idx, x in enumerate(pred_str):
            rr = compute_rouge([x], [reference], rouge_metric)
            rank_example['sampled']['scores'].append(rr)
            rouges.append(rr)
            r1s.append(rr['rouge1_f1'])
        best_idx = np.argmax(r1s)
        best_input = list(map(str, idxs[best_idx]))
        num_source_sent = len(best_input)
        rouge_df = pd.DataFrame(rouges)
        cols = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
        row = {
            'best_num_source_sents': num_source_sent, 'best_source_sent_idxs': ','.join(best_input),
            'num_source_sents': num_trunc_sent,
            'sent_compression': num_trunc_sent / num_source_sent
        }
        for col in cols:
            row[f'{col}_avg'] = rouge_df[col].mean()
            row[f'{col}_max'] = rouge_df[col].max()
            row[f'{col}_min'] = rouge_df[col].min()
        sample_out.append(row)
        rank_dataset.append(rank_example)

    sample_out = pd.DataFrame(sample_out)
    for col in list(sample_out.columns):
        try:
            print(col, sample_out[col].dropna().mean())
        except:
            print(col, ' not a valid column to average')

    out_fn = os.path.join(results_dir, f'{args.extractor}_sampled.csv')
    print(f'Saving to {out_fn}')
    sample_out.to_csv(out_fn, index=False)

    out_fn = os.path.join(results_dir, 'sample_dataset.json')
    print(f'Saving Rank dataset to {out_fn}...')
    with open(out_fn, 'w') as fd:
        ujson.dump(out_fn, fd)
