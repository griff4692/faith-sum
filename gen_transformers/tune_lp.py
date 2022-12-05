import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from collections import defaultdict

import argparse
import regex as re
from datasets import load_from_disk
import pandas as pd
import torch
import numpy as np
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import get_path_from_exp, infer_dataset
from eval.rouge_metric import RougeMetric
from gen_transformers.model import TransformerSummarizer
from gen_transformers.model_utils import sentence_indicators, infer_hf_model
from preprocess.extract_oracles import convert_to_sents
from preprocess.convert_abstractive_to_extractive import gain_selection


os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')
np.random.seed(1992)


# TODO: Grid-search
DATASET_KWARGS = {
    'cnn_dailymail': {
        'max_length': 142,
        'min_length': 56,
    },
    'samsum': {  # TODO idk
        'min_length': 10,
        'max_length': 100,
    },
    'xsum': {
        'min_length': 11,
        'max_length': 62,
    }
}


def compute_implied(args, nlp, pred_str, source_annotated):
    implied_extracts = []

    source_sents = re.split(r'<s\d*>', source_annotated)
    source_sents = [x.strip() for x in source_sents if len(x.strip()) > 0]

    source_sent_toks = [
        [str(token.text) for token in nlp(sentence)] for sentence in source_sents
    ]

    pred_sents = [
        convert_to_sents(x, nlp, is_dialogue=args.dataset == 'samsum')
        for x in pred_str
    ]
    num_pred = len(pred_str)
    pred_toks = [
        [[str(token.text) for token in sentence] for sentence in pred_sent]
        for pred_sent in pred_sents
    ]

    for idx in range(num_pred):
        implied_oracle_idx = gain_selection(
            source_sent_toks, pred_toks[idx], 5, lower=True, sort=True
        )[0]
        implied_oracle = ' '.join([str(source_sents[i]) for i in implied_oracle_idx])
        implied_extracts.append({
            'idxs': implied_oracle_idx,
            'summary': implied_oracle,
        })
    return implied_extracts


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


def gen_from_guide(
        args, model, tokenizer, source_annotated, idx_to_keep, special_id_min, fixed_length_penalty):
    has_bos = 'pegasus' not in args.hf_model

    source_annotated_rep = [source_annotated for _ in range(len(idx_to_keep))]

    inputs = tokenizer(
        source_annotated_rep,
        padding='longest',
        truncation=True,
        max_length=512 if 'pegasus' in args.hf_model else 1024,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(args.gpu_device)
    attention_mask = inputs['attention_mask'].to(args.gpu_device)
    cls_mask = input_ids >= special_id_min
    extract_indicators = []
    for cand_idx, extract_idx in enumerate(idx_to_keep):
        ei = sentence_indicators(
            cls_mask[cand_idx].unsqueeze(0), extract_idx, attention_mask[cand_idx].unsqueeze(0), has_bos=has_bos
        )
        extract_indicators.append(ei)
    extract_indicators = torch.cat(extract_indicators, dim=0)
    encoder_outputs = model.model.model.encoder(**{
        'input_ids': input_ids, 'attention_mask': attention_mask, 'extract_indicators': extract_indicators,
    })

    shared_kwargs = {
        'encoder_outputs': encoder_outputs,
        'attention_mask': attention_mask,
        'num_return_sequences': 1,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    }
    gen_kwargs = {
        'num_beams': 4 if args.dataset == 'cnn_dailymail' else 8,
    }

    shared_kwargs.update(gen_kwargs)
    shared_kwargs.update(DATASET_KWARGS[args.dataset])
    shared_kwargs['length_penalty'] = fixed_length_penalty

    with torch.no_grad(), torch.cuda.amp.autocast():
        pred_ids = model.model.generate(**shared_kwargs).tolist()

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    cand_metrics, best_metric, avg_r1_f1, diversity = model.score_candidates(
        [reference], pred_str, prefix='from_extract', eval=True,
    )
    return [y['best_from_extract_rouge1_f1'] for y in cand_metrics]


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
    parser.add_argument('--abstract_experiment', default='samsum_from_bert_red_extract_w_unlike')
    parser.add_argument('--extract_experiment', default='samsum_bert_red_extract_generator_3e5lr')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-do_not_save', default=False, action='store_true')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--max_examples', default=99999999, type=int)
    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--decode_method', default='beam', choices=['diverse', 'beam', 'nucleus'])
    parser.add_argument('--num_candidates', default=16, type=int)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--chunk', default=None)

    args = parser.parse_args()

    infer_hf_model(args)

    results_dir = os.path.join(args.data_dir, 'results', args.extract_experiment)
    decode_suffix = args.decode_method + '_' + str(args.num_candidates)
    chunk_suffix = '' if args.chunk is None else f'_chunk_{args.chunk}'
    in_fn = os.path.join(results_dir, f'{args.split}_{decode_suffix}_outputs{chunk_suffix}.csv')

    print(f'Reading in extracts from {in_fn}')
    outputs = pd.read_csv(in_fn)
    prev_n = len(outputs)
    outputs.dropna(subset=['extract'], inplace=True)
    n = len(outputs)
    if prev_n > n:
        print(f'Filtered out {prev_n - n} null extracts.')
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=111)
        n = len(outputs)

    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)[args.split]
    dataset_idx2id = dataset['id']
    all_source_annotated = dataset['source_annotated']

    records = outputs.to_dict('records')
    weight_dir = os.path.join(args.data_dir, 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.abstract_experiment)
    tokenizer_dir = os.path.join(weight_dir, args.abstract_experiment, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    additional_ids = tokenizer.additional_special_tokens_ids
    special_id_min = 999999 if len(additional_ids) == 0 else min(tokenizer.additional_special_tokens_ids)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(args.gpu_device).eval()

    if 'pegasus' not in args.hf_model:
        model = model.half()

    rouge_metric = RougeMetric()
    results = []
    lp_candidates = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    for lp in lp_candidates:
        stats_by_extract_len = defaultdict(list)
        for record in tqdm(records, total=len(records)):
            source_annotated = all_source_annotated[record['dataset_idx']]
            # Get source tokens
            reference = record['reference']
            extract_idx = [get_extract_idxs_from_str(x) for x in record['extract_idx'].split('<cand>')]
            extract_lens = [len(set(x)) for x in extract_idx]
            rouge1_f1s = gen_from_guide(
                args, model, tokenizer, source_annotated, extract_idx, special_id_min,
                fixed_length_penalty=lp
            )

            for r1, elen in zip(rouge1_f1s, extract_lens):
                stats_by_extract_len[elen].append(r1)

        for elen, r1s in stats_by_extract_len.items():
            row = {'extract_length': elen, 'rouge1': float(np.mean(r1s)), 'length_penalty': lp}
            results.append(row)
    results = pd.DataFrame(results)
    print(results.rouge1.tolist())
    out_fn = os.path.join(results_dir, 'tuned_extract_lengths.csv')
    print(f'Saving Results to {out_fn}')
    # results.to_csv(out_fn, index=False)

    for elen in sorted(results['extract_length'].unique().tolist()):
        er = results[results['extract_length'] == elen]
        best_row = er.rouge1.argmax()
        best_lp = er.length_penalty.tolist()[best_row]
        print(f'Best for {elen} length extracts -> {best_lp}')
