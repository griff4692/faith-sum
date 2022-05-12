import itertools
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import spacy
import numpy as np
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from data_utils import get_path_from_exp
from eval.rouge_metric import RougeMetric
from preprocess.convert_abstractive_to_extractive import gain_selection
from gen_transformers.model import TransformerSummarizer
from preprocess.extract_oracles import convert_to_sents
from datasets import load_dataset

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


def sentence_mask(cls_mask, sent_idx_to_mask):
    sent_mask = torch.zeros_like(cls_mask, device=cls_mask.device).long()
    sent_locs = cls_mask.nonzero()[:, 1]
    num_sents = len(sent_locs)
    for sent_idx, sent_loc in enumerate(sent_locs):
        sent_loc = sent_loc.item()
        end_loc = sent_locs[sent_idx + 1].item() if sent_idx + 1 < num_sents else len(sent_locs)
        if sent_idx in sent_idx_to_mask:
            sent_mask[0, sent_loc:end_loc] = 1
    return sent_mask


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
    return np.argsort(- avg_rs)


if __name__ == '__main__':
    extractor = 'abstract'
    oracle_df = pd.read_csv('/nlp/projects/faithsum/cnn_dailymail/oracle/validation_v2.csv')
    outputs = pd.read_csv('/nlp/projects/faithsum/results/score_abstract_v2/validation_beam_outputs.csv')

    nlp = spacy.load('en_core_web_sm')

    dataset = load_dataset('cnn_dailymail', '3.0.0')['validation']
    dataset_idx2id = dataset['id']
    orig_sources = dataset['article']

    gpu = 0
    data_dir = '/nlp/projects/faithsum'
    wandb_name = 'score_abstract_v2'
    hf_model = 'facebook/bart-base'

    ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}
    records = outputs.to_dict('records')
    weight_dir = os.path.join(data_dir, 'weights')

    ckpt_path = get_path_from_exp(weight_dir, wandb_name)
    tokenizer_dir = os.path.join(weight_dir, wandb_name, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    additional_ids = tokenizer.additional_special_tokens_ids
    special_id_min = 999999 if len(additional_ids) == 0 else min(tokenizer.additional_special_tokens_ids)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=hf_model, strict=False).to(gpu).eval()

    target_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    rouge_metric = RougeMetric()
    updated_records = []
    n = len(records)
    correlations = []
    for record in tqdm(records, total=n):
        oracle_obj = ids2oracles[dataset_idx2id[record['dataset_idx']]]
        source = orig_sources[record['dataset_idx']]
        source_sents = convert_to_sents(source, nlp)
        n = len(source_sents)

        sent_scores = np.array(list(map(float, record['sent_scores'].split(','))))
        num_trunc_sent = len(sent_scores)  # Up to 1024 tokens usually sometimes reduces number of sentences
        assert num_trunc_sent <= len(source_sents)
        source_sents = source_sents[:num_trunc_sent]

        source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Get source tokens
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        pred_abstract = record['abstract']
        reference = record['reference']

        extract_priority = (-sent_scores).argsort()
        oracle_priority = get_priority(source_sents_tok, reference, nlp)
        abstract_priority = get_priority(source_sents_tok, pred_abstract, nlp)

        oc_ex = pearsonr(oracle_priority, extract_priority)[0]
        oc_abs = pearsonr(oracle_priority, abstract_priority)[0]
        ex_abs = pearsonr(extract_priority, abstract_priority)[0]
        correlations.append({'oracle_extract': oc_ex, 'oracle_abstract': oc_abs, 'extract_abstract': ex_abs})

        if extractor == 'abstract':
            priority = abstract_priority
        elif extractor == 'extract':
            priority = extract_priority
        elif extractor == 'oracle':
            priority = oracle_priority

        inputs = tokenizer(
            [source_annotated],
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].to(gpu)
        attention_mask = inputs['attention_mask'].to(gpu)
        cls_mask = input_ids >= special_id_min

        idxs = []
        masks = []
        for length in target_lengths:
            end_range = min(n, length)
            unmask_idxs = list(sorted(priority[:end_range]))
            idxs.append(unmask_idxs)
            masks.append(sentence_mask(cls_mask, unmask_idxs))

        order = [
            'extract',
            'implied',
            'oracle',
            'oracle_aligned',
            'abstract_aligned',
        ]

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

        row = {}
        for idx, x in enumerate(pred_str):
            row[f'from_{target_lengths[idx]}'] = x
            row.update(compute_rouge([x], [reference], rouge_metric, prefix=f'from_{target_lengths[idx]}_'))

        all_r1s = [v for k, v in row.items() if 'rouge1_f1' in k]
        best_r1 = max(all_r1s)
        row['best_rouge1_f1'] = best_r1

        record.update(row)
        updated_records.append(record)
    updated_df = pd.DataFrame(updated_records)
    out_fn = 'updated.csv'
    updated_df.to_csv(out_fn, index=False)

    cols_to_print = [
        f'from_{l}_rouge1_f1' for l in target_lengths
    ]
    cols_to_print.append('best_rouge1_f1')
    for col in cols_to_print:
        print(updated_df[col].mean())

    correlations = pd.DataFrame(correlations)
    for col in correlations.columns:
        print(f'{col} -> {correlations[col].mean()}')
