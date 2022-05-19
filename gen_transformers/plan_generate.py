import itertools
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import spacy
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

from data_utils import get_path_from_exp
from eval.rouge_metric import RougeMetric
from preprocess.convert_abstractive_to_extractive import gain_selection
from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import sentence_mask
from preprocess.extract_oracles import convert_to_sents


os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


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


if __name__ == '__main__':
    oracle_df = pd.read_csv('/nlp/projects/faithsum/cnn_dailymail/oracle/validation_v2.csv')
    outputs = pd.read_csv('/nlp/projects/faithsum/results/score_abstract_v2/validation_beam_outputs.csv')
    # outputs = outputs.sample(n=20, replace=False)

    nlp = spacy.load('en_core_web_sm')

    dataset = load_dataset('cnn_dailymail', '3.0.0')['validation']
    dataset_idx2id = dataset['id']

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

    rouge_metric = RougeMetric()
    updated_records = []
    n = len(records)
    # for record in tqdm(records, total=n):
    for record in records:
        oracle_obj = ids2oracles[dataset_idx2id[record['dataset_idx']]]
        source = record['source']
        pred_abstract = record['abstract']
        reference = record['reference']
        extract_idx = get_idx(record['extract_idx'])
        implied_idx = get_idx(record['implied_extract_idx'])
        oracle_idx = get_idx(oracle_obj['sent_idxs'])

        source_sents = convert_to_sents(source, nlp)
        source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Get source tokens
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]

        oracle_aligned_sents, oracle_aligned_sents_flat = get_alignments(source_sents_tok, reference, nlp)
        abstract_aligned_sents, abstract_aligned_sents_flat = get_alignments(source_sents_tok, pred_abstract, nlp)

        # inputs = tokenizer(
        #     [source_annotated],
        #     padding='longest',
        #     truncation=True,
        #     max_length=1024,
        #     return_tensors='pt',
        # )
        # input_ids = inputs['input_ids'].to(gpu)
        # attention_mask = inputs['attention_mask'].to(gpu)
        # cls_mask = input_ids >= special_id_min
        #
        # extract_mask = sentence_mask(cls_mask, extract_idx)
        # implied_mask = sentence_mask(cls_mask, implied_idx)
        # oracle_mask = sentence_mask(cls_mask, oracle_idx)
        # oracle_aligned_mask = sentence_mask(cls_mask, oracle_aligned_sents_flat)
        # abstract_aligned_mask = sentence_mask(cls_mask, abstract_aligned_sents_flat)

        inputs_filt = [
            ' '.join([str(source_sents[i]) for i in extract_idx if i < len(source_sents)]),
            ' '.join([str(source_sents[i]) for i in implied_idx if i < len(source_sents)]),
            ' '.join([str(source_sents[i]) for i in oracle_idx if i < len(source_sents)]),
            ' '.join([str(source_sents[i]) for i in oracle_aligned_sents_flat if i < len(source_sents)]),
            ' '.join([str(source_sents[i]) for i in abstract_aligned_sents_flat if i < len(source_sents)]),
        ]

        inputs = tokenizer(
            inputs_filt,
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].to(gpu)
        attention_mask = inputs['attention_mask'].to(gpu)

        order = [
            'extract',
            'implied',
            'oracle',
            'oracle_aligned',
            'abstract_aligned',
        ]

        # all_masks = torch.cat([
        #     extract_mask,
        #     implied_mask,
        #     oracle_mask,
        #     oracle_aligned_mask,
        #     abstract_aligned_mask,
        # ])

        # n = len(all_masks)

        # encoder_inputs = model.model.model.encoder(input_ids=input_ids).last_hidden_state
        # input_ids_rep = input_ids.repeat(n, 1)
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
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
        # extract_mask_pred, implied_mask_pred, oracle_mask_pred = pred_str

        oracle_sum = ' '.join([str(source_sents[i]) for i in oracle_idx if i < len(source_sents)])

        row = {}
        for idx, x in enumerate(pred_str):
            row[f'from_{order[idx]}'] = x
            row.update(compute_rouge([x], [reference], rouge_metric, prefix=f'from_{order[idx]}_'))

        # print('\n')
        # print(
        #     record['rouge1_f1'], row['from_extract_rouge1_f1'], row['from_implied_rouge1_f1'],
        #     row['from_oracle_rouge1_f1'], row['from_oracle_aligned_rouge1_f1'], row['from_abstract_aligned_rouge1_f1'],
        # )

        print('Reference:')
        print(reference)
        print('Abstract:')
        print(pred_abstract)
        print('Available Sentences:')
        print(oracle_sum)
        print('Predicted Summary:')
        print(pred_str[2])
        print('\n')
        print('-' * 100)
        print('\n')

        record.update(row)
        updated_records.append(record)
    updated_df = pd.DataFrame(updated_records)
    out_fn = 'updated.csv'
    updated_df.to_csv(out_fn, index=False)

    cols_to_print = [
        'rouge1_f1', 'from_extract_rouge1_f1', 'from_implied_rouge1_f1', 'from_oracle_rouge1_f1',
        'from_oracle_aligned_rouge1_f1', 'from_abstract_aligned_rouge1_f1'
    ]
    for col in cols_to_print:
        print(updated_df[col].mean())
