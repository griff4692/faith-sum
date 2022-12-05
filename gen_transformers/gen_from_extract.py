import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
        'length_penalty': 1.0,  # previously was 4.0, could try 1.0
        'max_length': 142,
        'min_length': 56,
    },
    'samsum': {  # TODO idk
        'min_length': 10,
        'max_length': 100,
        'length_penalty': [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
    },
    'xsum': {
        'min_length': 11,
        'max_length': 62,
        'length_penalty': 0.6
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


def gen_from_guide(args, nlp, model, tokenizer, source_annotated, idx_to_keep, special_id_min, num_return_sequences=1):
    has_bos = 'pegasus' not in args.hf_model
    n = len(idx_to_keep)

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

    # No highlights on last beam
    if args.convert_last_to_unprompted:
        # Highlights every sentence and lets model choose...
        idx_to_keep[-1] = list(range(100))

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
        'num_return_sequences': num_return_sequences,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    }
    if num_return_sequences == 1:
        gen_kwargs = {
            'num_beams': 4 if args.dataset == 'cnn_dailymail' else 8,
        }
    else:
        gen_kwargs = {
            # 'diversity_penalty': 1.0,
            'num_beams': num_return_sequences,
            # 'num_beam_groups': num_return_sequences,
        }

    shared_kwargs.update(gen_kwargs)
    shared_kwargs.update(DATASET_KWARGS[args.dataset])
    if type(shared_kwargs['length_penalty']) == list:
        lp_arr = shared_kwargs['length_penalty']
        lp = []
        for idx in range(n):
            elen = len(set(idx_to_keep[idx]))
            dynamic_lp = lp_arr[min(elen, len(lp_arr)) - 1]
            lp.append(dynamic_lp)
        shared_kwargs['length_penalty'] = lp

    with torch.no_grad(), torch.cuda.amp.autocast():
        pred_ids = model.model.generate(**shared_kwargs).tolist()

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    implied_extracts = compute_implied(args, nlp, pred_str, source_annotated)

    ps, rs, f1s = [], [], []
    for extract_idx, implied in zip(idx_to_keep, implied_extracts):
        if len(extract_idx) == 0 or len(extract_idx) == 100:
            assert args.convert_last_to_unprompted
            continue

        implied_idx = implied['idxs']
        agreement = set(extract_idx).intersection(implied_idx)
        n = len(agreement)
        r = n / len(extract_idx)
        p = n / len(implied_idx)
        f1 = 0 if min(r, p) == 0 else (2 * p * r) / (p + r)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    mean_p = np.mean(ps)
    mean_r = np.mean(rs)
    mean_f1 = np.mean(f1s)

    cand_metrics, best_metric, avg_r1_f1, diversity = model.score_candidates(
        [reference], pred_str, prefix='from_extract', eval=True,
    )
    rouge1_f1s = [y['best_from_extract_rouge1_f1'] for y in cand_metrics]
    pred_rank = int(np.argmax(rouge1_f1s))
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
        'plan_precision': mean_p,
        'plan_recall': mean_r,
        'plan_f1': mean_f1,
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
    parser.add_argument('--abstract_experiment', default='extract_indicators')
    parser.add_argument('--extract_experiment', default='add_doc')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-do_not_save', default=False, action='store_true')
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--max_examples', default=999999, type=int)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--decode_method', default='beam', choices=['diverse', 'beam', 'nucleus'])
    parser.add_argument('--num_candidates', default=16, type=int)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--split', default='test')
    parser.add_argument('-verbose', default=False, action='store_true')
    parser.add_argument('-add_abstract_experiment', default=False, action='store_true')
    parser.add_argument('-convert_last_to_unprompted', default=False, action='store_true')
    parser.add_argument('--chunk', default=None)

    args = parser.parse_args()

    infer_dataset(args, 'extract_experiment')
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
    df = []
    updated_records = []
    stats = []
    wins, losses, ties = 0, 0, 0
    compare_col = 'best_extract_rouge1_f1' if args.num_candidates > 1 else 'extract_rouge1_f1'

    nlp = spacy.load('en_core_web_sm')

    for record in tqdm(records, total=len(records)):
        source_annotated = all_source_annotated[record['dataset_idx']]
        # Get source tokens
        reference = record['reference']

        extract_idx = [get_extract_idxs_from_str(x) for x in record['extract_idx'].split('<cand>')]
        if args.top_k is not None and args.top_k < len(extract_idx):
            extract_idx = extract_idx[:args.top_k]
        gen_output = gen_from_guide(
            args, nlp, model, tokenizer, source_annotated, extract_idx, special_id_min,
            num_return_sequences=args.num_return_sequences
        )

        best_ensemble_rouge1_f1 = max(record[compare_col], gen_output['best_from_extract_rouge1_f1'])
        stat_row = {
            compare_col: record[compare_col],
            'best_from_extract_rouge1_f1': gen_output['best_from_extract_rouge1_f1'],
            'best_ensemble_rouge1_f1': best_ensemble_rouge1_f1, 'plan_f1': gen_output['plan_f1']
        }

        if args.num_candidates > 1:
            more = {
                'avg_extract_rouge1_f1': record['avg_extract_rouge1_f1'],
                'avg_from_extract_rouge1_f1': gen_output['avg_from_extract_rouge1_f1'],
                'min_from_extract_rouge1_f1': gen_output['min_from_extract_rouge1_f1'],
                'diversity': gen_output['diversity']
            }
            stat_row.update(more)
        stats.append(stat_row)

        if gen_output['best_from_extract_rouge1_f1'] >= record[compare_col]:
            wins += 1
            gen_output['abstract_win'] = 1
        elif gen_output['best_from_extract_rouge1_f1'] == record[compare_col]:
            ties += 1
            gen_output['abstract_win'] = 0  # Treat as a loss for lower-bound
        else:
            losses += 1
            gen_output['abstract_win'] = 0
        gen_output['best_ensemble_rouge1_f1'] = best_ensemble_rouge1_f1
        record.update(gen_output)
        updated_records.append(record)
        if args.verbose:
            print(f'Abstract Wins: {wins}. Losses: {losses}. Ties: {ties}')
    stats = pd.DataFrame(stats)
    avgs = {k: stats[k].mean() for k in stats.columns}
    print(avgs)
    winshare = avgs['best_from_extract_rouge1_f1'] - avgs[compare_col]
    print(f'Win share: {winshare}')

    updated_df = pd.DataFrame(updated_records)
    if not args.do_not_save:
        top_k_str = '' if args.top_k is None else f'_{args.top_k}'
        chunk_suffix = '' if args.chunk is None else f'_chunk_{args.chunk}'
        prompt_suffix = '_w_unprompted' if args.convert_last_to_unprompted else ''
        if args.add_abstract_experiment:
            out_fn = os.path.join(
                results_dir,
                f'{args.split}_from_{decode_suffix}_extract{top_k_str}_{args.abstract_experiment}{prompt_suffix}{chunk_suffix}.csv'
            )
        else:
            out_fn = os.path.join(
                results_dir,
                f'{args.split}_from_{decode_suffix}_extract{top_k_str}{prompt_suffix}{chunk_suffix}.csv'
            )
        print(f'Saving prompted abstracts to {out_fn}')
        updated_df.to_csv(out_fn, index=False)

    extract_tok_len = updated_df['extract'].apply(lambda x: len(
        x.split('<cand>')[0].split(' '))).mean()
    ref_tok_len = updated_df['reference'].apply(
        lambda x: len(x.split(' '))).mean()
    from_extract_tok_len = updated_df['from_extract_abstract'].apply(
        lambda x: len(x.split('<cand>')[0].split(' '))).mean()

    print(f'Average Extract Tokens: {extract_tok_len}')
    print(f'Average From Extract Abstract Tokens: {from_extract_tok_len}')
    print(f'Average Reference Tokens: {ref_tok_len}')
    avg_win = updated_df['abstract_win'].mean() / len(updated_df)
    print(f'Fraction of From Extract Wins over Extract: {avg_win}')
