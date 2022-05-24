import itertools
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import argparse
import spacy
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from scipy.special import expit, softmax

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


def gen_from_mask(model, tokenizer, source_annotated, idx_to_keep, special_id_min):
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
    updated_mask = sentence_mask(cls_mask, idx_to_keep, attention_mask)
    kwargs = {
        'input_ids': input_ids,
        'attention_mask': updated_mask,
        'num_return_sequences': 1,
        'num_beams': 4,
        'length_penalty': 4.0,
        'max_length': 142,
        'min_length': 56,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    }

    pred_ids = model.model.generate(**kwargs)
    pred_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)[0]
    rr = compute_rouge([pred_str], [reference], rouge_metric)
    rr['prediction'] = pred_str
    return rr


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample from Top-K')

    parser.add_argument('--extractor', default='extract', choices=['oracle', 'extract'])
    parser.add_argument('--gpu_device', default=1, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='select_extract_abstract')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--hf_model', default='facebook/bart-base')
    parser.add_argument('--max_examples', default=25, type=int)
    parser.add_argument('--extract_sent_len', default=3, type=int)
    parser.add_argument('--samples', default=16, type=int)
    parser.add_argument('-use_scores', default=False, action='store_true')
    parser.add_argument('--k', default=10, type=int)

    args = parser.parse_args()

    oracle_df = pd.read_csv(os.path.join(args.data_dir, 'cnn_dailymail/oracle/validation_v2.csv'))
    results_dir = os.path.join(args.data_dir, 'results', args.wandb_name)
    outputs = pd.read_csv(os.path.join(results_dir, 'validation_beam_outputs.csv'))
    n = len(outputs)
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=1992)
        n = len(outputs)
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
    df = []
    updated_records = []
    stats = []
    for record in tqdm(records, total=len(records)):
        oracle_obj = ids2oracles[dataset_idx2id[record['dataset_idx']]]
        source = orig_sources[record['dataset_idx']]
        source_sents = convert_to_sents(source, nlp)
        sent_scores = np.array(list(map(float, record['sent_scores'].split(','))))
        num_trunc_sent = len(sent_scores)  # Up to 1024 tokens usually sometimes reduces number of sentences
        assert num_trunc_sent <= len(source_sents)
        source_sents = source_sents[:num_trunc_sent]

        implied_idx = list(map(int, record['implied_extract_idx'].split(',')))
        # for idx in implied_idx:
        #     sent_scores[idx] += 0.25

        source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Get source tokens
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        pred_abstract = record['abstract']
        reference = record['reference']

        extract_priority = (-sent_scores).argsort()
        extract_scores = sent_scores[extract_priority]
        extract_scores_norm = expit(extract_scores)

        if args.extractor == 'extract':
            priority = extract_priority
            scores = extract_scores
        elif args.extractor == 'oracle':
            oracle_priority, oracle_scores = get_priority(source_sents_tok, reference, nlp)
            priority = oracle_priority
            scores = oracle_scores
        else:
            raise Exception('Unknown')

        n = len(priority)
        trunc_idx = min(n, args.k)
        top_k_priority = priority[:trunc_idx]
        p = softmax(scores[:trunc_idx]) if args.use_scores else None
        # topk_idx = top_k_priority[:min(n, args.extract_sent_len)]
        # To use the bottom is to incorporate trigram blocking
        topk_idx = list(map(int, record['extract_idx'].split(',')))
        size = min(n, args.extract_sent_len)

        sampled_idxs = []
        extracts = []
        for _ in range(args.samples):
            sampled_idx = list(np.random.choice(top_k_priority, size=size, replace=False, p=p))
            sampled_idxs.append(sampled_idx)
            extracts.append(' '.join([str(source_sents[i]) for i in sampled_idx]))
        rouges = []
        r1s = []
        for idx, x in enumerate(extracts):
            rr = compute_rouge([x], [reference], rouge_metric)
            rouges.append(rr)
            r1s.append(rr['rouge1_f1'])

        mean_r1 = np.mean(r1s)
        min_r1 = np.min(r1s)
        max_r1 = np.max(r1s)

        gen_output = gen_from_mask(model, tokenizer, source_annotated, topk_idx, special_id_min)

        stats.append({
            'mean_f1': mean_r1,
            'min_r1': min_r1,
            'max_r1': max_r1,
            'extract_rouge1_f1': record['extract_rouge1_f1'],
            'prompt_abstract_rouge1_f1': gen_output['rouge1_f1'],
            'abstract_rouge1_f1': record['rouge1_f1'],
        })
        record['from_extract_rouge1_f1'] = gen_output['rouge1_f1']
        record['from_extract_rouge1_recall'] = gen_output['rouge1_recall']
        record['from_extract_rouge1_precision'] = gen_output['rouge1_precision']
        record['from_extract_rouge2_f1'] = gen_output['rouge2_f1']
        record['from_extract_rouge2_recall'] = gen_output['rouge2_recall']
        record['from_extract_rouge2_precision'] = gen_output['rouge2_precision']
        record['from_extract_rougeL_f1'] = gen_output['rougeL_f1']
        record['from_extract_rougeL_recall'] = gen_output['rougeL_recall']
        record['from_extract_rougeL_precision'] = gen_output['rougeL_precision']
        record['from_extract_abstract'] = gen_output['prediction']
        updated_records.append(record)

    stats = pd.DataFrame(stats)
    avgs = {k: stats[k].mean() for k in stats.columns}
    avgs['k'] = args.k
    avgs['use_scores'] = args.use_scores
    avgs['samples'] = args.samples
    print(avgs)

    updated_df = pd.DataFrame(updated_records)
    updated_out_fn = os.path.join(results_dir, 'from_extract.csv')
    print(f'Saving prompted abstracts to {updated_out_fn}')
    updated_df.to_csv(updated_out_fn, index=False)

    abstract_tok_len = updated_df['abstract'].apply(lambda x: len(x.split(' '))).mean()
    extract_tok_len = updated_df['extract'].apply(lambda x: len(x.split(' '))).mean()
    ref_tok_len = updated_df['reference'].apply(lambda x: len(x.split(' '))).mean()
    from_extract_tok_len = updated_df['from_extract_abstract'].apply(lambda x: len(x.split(' '))).mean()

    print(f'Average Abstract Tokens: {abstract_tok_len}')
    print(f'Average Extract Tokens: {extract_tok_len}')
    print(f'Average From Extract Abstract Tokens: {from_extract_tok_len}')
    print(f'Average Reference Tokens: {ref_tok_len}')
