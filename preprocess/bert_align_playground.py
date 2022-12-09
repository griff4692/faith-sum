import itertools
import os
import string

import pandas as pd
import regex as re

import argparse
import spacy
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
from preprocess.convert_abstractive_to_extractive import gain_selection
from sklearn.metrics.pairwise import cosine_similarity
from bert_score.scorer import BERTScorer
from nltk import word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoModel, AutoTokenizer

STOPWORDS = set([x.lower() for x in stopwords.words('english')])

from sum_constants import summarization_name_mapping


def bs_overlap(bs, aligned_idxs, annotations, source_sents):
    annotation_sents = [
        ' '.join([source_sents[i] for i in annot]) for annot in annotations
    ]

    aligned_sents = [source_sents[i] for i in aligned_idxs]
    aligned_rep = [' '.join(aligned_sents) for _ in range(len(annotations))]
    bs_p, bs_r, bs_f1 = bs.score(aligned_rep, annotation_sents)
    return float(bs_f1.mean().item())


def tune_coverage(
        args, model, tokenizer, bs, avg_thresholds, max_imp_thresholds, max_coverages, max_retrieval_candidates
):
    outputs = []
    best_bs = 0
    for avg_imp_threshold in avg_thresholds:
        avg_imp_threshold = float(avg_imp_threshold)
        for max_imp_threshold in max_imp_thresholds:
            max_imp_threshold = float(max_imp_threshold)
            for max_coverage in max_coverages:
                for max_retrievals in max_retrieval_candidates:
                    ps, rs, f1s = [], [], []
                    lens = []
                    bs_align = []
                    for example in tqdm(examples):
                        annotations = example['annotations']
                        align_idxs, aligned_sents = add_bert_alignment_no_red(
                            model, tokenizer, example['dialogue'], example['summary'],
                            avg_imp_threshold=avg_imp_threshold,
                            max_imp_threshold=max_imp_threshold,
                            max_coverage=max_coverage,
                            max_retrievals=max_retrievals
                        )

                        bs_align.append(
                            bs_overlap(bs, align_idxs, annotations, example['dialogue'])
                        )

                        lens.append(len(align_idxs))

                        p, r, f1 = compute_alignment_score(align_idxs, annotations)
                        ps.append(p)
                        rs.append(r)
                        f1s.append(f1)

                    old_best = best_bs
                    avg_f1 = float(np.mean(f1s))
                    avg_bs = float(np.mean(bs_align))
                    best_bs = max(avg_bs, best_bs)
                    if old_best != best_bs:
                        print('\n', best_bs)

                    outputs.append({
                        'avg_imp_threshold': avg_imp_threshold,
                        'avg_max_threshold': max_imp_threshold,
                        'max_coverage': max_coverage,
                        'max_retrievals': max_retrievals,
                        'p': float(np.mean(ps)),
                        'r': float(np.mean(rs)),
                        'f1': avg_f1,
                        'bs_align': avg_bs,
                        'length': float(np.mean(lens)),
                    })
    outputs = pd.DataFrame(outputs)
    print(args.dataset + ' coverage tuning results...')
    outputs = outputs.sort_values(by='f1', ascending=False).reset_index(drop=True)
    print(outputs.head(n=5))

    outputs = outputs.sort_values(by='bs_align', ascending=False).reset_index(drop=True)
    print(outputs.head(n=5))
    outputs.to_csv(f'{args.dataset}_bert_coverage_tune_results.csv', index=False)


def tune_align(args, bs, thresholds, max_retrieval_candidates, p_factors):
    outputs = []
    best_bs = 0
    for threshold in thresholds:
        for max_retrievals in max_retrieval_candidates:
            for p_factor in p_factors:
                ps, rs, f1s = [], [], []
                lens = []
                bs_align = []
                for example in tqdm(examples):
                    annotations = example['annotations']
                    align_idxs, aligned_sents = add_bert_alignment(
                        bs, example['dialogue'], example['summary'],
                        threshold=threshold,
                        p_factor=p_factor,
                        max_per_sent=max_retrievals
                    )

                    bs_align.append(
                        bs_overlap(bs, align_idxs, annotations, example['dialogue'])
                    )

                    lens.append(len(align_idxs))

                    p, r, f1 = compute_alignment_score(align_idxs, annotations)
                    ps.append(p)
                    rs.append(r)
                    f1s.append(f1)

                    old_best = best_bs
                    avg_f1 = float(np.mean(f1s))
                    avg_bs = float(np.mean(bs_align))
                    best_bs = max(avg_bs, best_bs)
                    if old_best != best_bs:
                        print('\n', best_bs)

                    outputs.append({
                        'threshold': threshold,
                        'max_retrievals': max_retrievals,
                        'p_factor': p_factor,
                        'p': float(np.mean(ps)),
                        'r': float(np.mean(rs)),
                        'f1': avg_f1,
                        'bs_align': avg_bs,
                        'length': float(np.mean(lens)),
                    })

    outputs = pd.DataFrame(outputs)
    print(args.dataset + ' coverage tuning results...')
    outputs = outputs.sort_values(by='f1', ascending=False).reset_index(drop=True)
    print(outputs.head(n=5))

    outputs = outputs.sort_values(by='bs_align', ascending=False).reset_index(drop=True)
    print(outputs.head(n=5))
    outputs.to_csv(f'{args.dataset}_bert_align_tune_results.csv', index=False)


def encode(text, model, tokenizer, device, max_length=128, max_batch_size=100, top_n_layers=1):
    seq_lens = []
    outputs = {'hidden_states': [], 'logits': []}
    text_batches = [list(x) for x in np.array_split(np.arange(len(text)), round(len(text) // max_batch_size) + 1)]
    text_batches = [x for x in text_batches if len(x) > 0]
    inputs = tokenizer(text, truncation=True, padding='longest', max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        for batch_idxs in text_batches:
            batch_inputs = {k: v[batch_idxs].to(device) for k, v in inputs.items()}
            seq_lens += batch_inputs['attention_mask'].sum(dim=1).tolist()
            batch_output = model(**batch_inputs, output_hidden_states=True)
            h_pool = torch.stack(batch_output['hidden_states'][-top_n_layers:]).mean(dim=0).cpu()
            outputs['hidden_states'].append(h_pool)
            if 'logits' in batch_output:
                logits = batch_output['logits'].cpu()
                outputs['logits'].append(logits)
    outputs['hidden_states'] = torch.cat(outputs['hidden_states'], dim=0)
    if len(outputs['logits']) > 0:
        outputs['logits'] = torch.cat(outputs['logits'], dim=0)
    return outputs, seq_lens


def compute_adj_f1(p, r, p_factor):
    if min(p, r) == 0:
        return 0
    return ((1 + p_factor ** 2) * p * r) / ((p_factor ** 2 * p) + r)


def retrieve_context(source_sents, source_hs, target_sent, target_h, max_retrievals, max_coverage):
    added_sents = []
    added_sent_idxs = []
    added_priors = []
    added_scores = []
    target_n = min(len(source_sents), max_retrievals)
    used_ids = []
    improvements = []
    max_coverages = np.zeros(shape=(len(target_h), ))
    sims = [
        np.clip(cosine_similarity(target_h, x).max(axis=1), 0, 1) for x in source_hs
    ]
    scores_prior = [sim.mean() for sim in sims]
    while len(added_sents) < target_n:
        weights = np.clip(1 - max_coverages, 1e-4, 1)
        # If we've covered a word more than 0.75, let's treat it as fully covered
        weights[weights < 1 - max_coverage] = 1e-4
        # Assign a low weight to stopwords
        # weights[stopword_mask] /= 2
        # No CLS or SEP token
        weights[0] = 0
        weights[-1] = 0

        # weights[np.where(weights <= 0.1)] = 0
        scores = [(sim * weights).sum() / weights.sum() for sim in sims]
        for id in used_ids:  # Don't reuse the same sentence
            scores[id] = float('-inf')
        max_score = np.max(scores)
        best_id = int(np.argmax(scores))
        used_ids.append(best_id)
        added_sent_idxs.append(best_id)
        added_sents.append(source_sents[best_id])
        new_max_coverages = np.maximum(max_coverages, sims[best_id])
        max_improvement = (new_max_coverages - max_coverages).max()
        mean_improvement = (new_max_coverages - max_coverages).mean()
        improvements.append((mean_improvement, max_improvement))
        added_priors.append(scores_prior[best_id])
        added_scores.append(max_score)
        max_coverages = new_max_coverages
    return added_sents, added_sent_idxs, max_coverages, added_scores, added_priors, improvements


def add_bert_alignment(bs, source_sents, target_sents, threshold=0.9, p_factor=1.0, max_per_sent=3):
    added_sents = []
    all_sents = set()
    # source_sents = [' '.join(word_tokenize(source_sent)) for source_sent in source_sents]
    # target_sents = [' '.join(word_tokenize(target_sent)) for target_sent in target_sents]
    assert len(target_sents) > 0
    for target_sent in target_sents:
        target_rep = [target_sent for _ in range(len(source_sents))]
        scores = bs.score(target_rep, source_sents)
        p = scores[0]
        # p_adj = [0 if i in all_sents else x for i, x in enumerate(p.tolist())]
        p_adj = p.tolist()
        r = scores[1]
        # r_adj = [0 if i in all_sents else x for i, x in enumerate(r.tolist())]
        r_adj = r.tolist()
        adj_f1 = [compute_adj_f1(p, r, p_factor) for (p, r) in zip(p_adj, r_adj)]
        # f1 = scores[-1]
        # f1_adj = [0 if i in all_sents else x for i, x in enumerate(f1.tolist())]
        keep_sents = [i for i in range(len(adj_f1)) if adj_f1[i] >= threshold]
        keep_scores = [adj_f1[i] for i in keep_sents]
        if len(keep_sents) == 0:
            # None over threshold take argmax
            keep_sents = [int(np.argmax(adj_f1))]
        elif len(keep_sents) > max_per_sent:
            priority = np.argsort(-np.array(keep_scores))
            filt = priority[:max_per_sent]
            keep_sents = [i for filt_idx, i in enumerate(keep_sents) if filt_idx in filt]
        for sent_idx in keep_sents:
            all_sents.add(sent_idx)
        added_sents.append(keep_sents)
    oracle_idxs = list(sorted(list(set(list(itertools.chain(*added_sents))))))
    assert len(oracle_idxs) > 0
    return oracle_idxs, added_sents


def compute_alignment_score(pred_idxs, annotation_idxs):
    rs = []
    ps = []
    f1s = []
    for annot in annotation_idxs:
        num_intersect = len(set(annot).intersection(set(pred_idxs)))
        r = num_intersect / len(annot)
        p = num_intersect / len(pred_idxs)
        f1 = 0 if min(r, p) == 0 else 2 * p * r / (p + r)
        rs.append(r)
        ps.append(p)
        f1s.append(f1)
    return np.mean(ps), np.mean(rs), np.mean(f1s)


def add_bert_alignment_no_red(
        model, tokenizer, source_sents, target_sents,
        avg_imp_threshold=0.01, max_imp_threshold=0.05, max_retrievals=3, max_coverage=0.85,
):
    source_hs, source_lens = encode(source_sents, model, tokenizer, model.device)
    target_hs, target_lens = encode(target_sents, model, tokenizer, model.device)

    source_hs_trunc = [
        source_h[:int(source_len)].numpy() for source_len, source_h in zip(source_lens, source_hs['hidden_states'])
    ]

    idxs = []
    keep_sents = []
    for t_idx, target_sent in enumerate(target_sents):
        target_len = target_lens[t_idx]
        target_h = target_hs['hidden_states'][t_idx][:target_len].numpy()
        added_sents, added_sent_idxs, max_coverages, added_scores, added_priors, improvements = retrieve_context(
            source_sents, source_hs_trunc, target_sent, target_h, max_coverage=max_coverage,
            max_retrievals=max_retrievals,
        )

        for sent, source_idx, imp in zip(added_sents, added_sent_idxs, improvements):
            if imp[0] >= avg_imp_threshold or imp[1] >= max_imp_threshold:
                keep_sents.append((t_idx, sent, imp))
                if source_idx not in idxs:
                    idxs.append(source_idx)
    return idxs, keep_sents


def read_annotations(annotation_fn, nlp):
    annots_len = []

    with open(annotation_fn, 'r') as fd:
        lines = fd.readlines()
        lines = [x.strip() for x in lines if len(x.strip()) > 0]

        examples = []
        for idx, line in enumerate(lines):
            if line == 'BEGIN_EXAMPLE':
                curr_dialogue = []
                curr_annotations = []
                curr_summary = []
            elif line.startswith('<s'):
                curr_dialogue.append(re.sub('<s\d+> ', '', line))
            elif line.startswith('ANNOTATION'):
                annots = [int(x.strip()) for x in line.split(':')[1].strip().split(',') if len(x.strip()) > 0]
                if len(annots) > 0:
                    annots_len.append(len(annots))
                    curr_annotations.append(annots)
            elif line == 'SUMMARY:':
                if args.dataset == 'samsum':
                    curr_summary = [str(x) for x in list(nlp(lines[idx + 1]).sents) if len(x) > 0]
                else:
                    curr_summary = []
                    for next in range(idx + 1, len(lines)):
                        curr_summary.append(str(lines[next]))
                        if lines[next + 1].startswith('ANNOTATION'):
                            break
            elif line == 'END_EXAMPLE':
                examples.append({
                    'dialogue': curr_dialogue,
                    'annotations': curr_annotations,
                    'summary': curr_summary,
                })
        return examples, annots_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='validation')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--mode', default='tune', choices=['tune', 'run'])
    parser.add_argument('--algorithm', default='bert_coverage', choices=['bert_coverage', 'bert_align', 'rouge'])
    parser.add_argument('-save_annotation_examples', default=False, action='store_true')

    args = parser.parse_args()

    if args.save_annotation_examples:
        input_col, target_col = summarization_name_mapping[args.dataset]
        out_dir = os.path.join(args.data_dir, args.dataset)
        dataset = load_from_disk(out_dir)
        print('Save annotation examples')
        val = dataset['validation']
        rand_idxs = list(np.sort(np.random.choice(np.arange(len(val)), size=(30, ), replace=False)))
        val_subset = val.select(rand_idxs)
        annotation_fn = os.path.expanduser(f'./analysis/{args.dataset}_annotations.txt')
        lines = []
        for record in val_subset:
            id = record['id']
            sa = re.sub(r'(<s\d+>)', r'\n\1', record['source_annotated']).strip()
            ref = record[target_col]
            lines += ['BEGIN_EXAMPLE', f'ID: {id}', 'DIALOGUE:']
            lines.append(sa)
            lines.append('SUMMARY:')
            lines.append(ref)
            lines.append('ANNOTATION 1: ')
            lines.append('ANNOTATION 2: ')
            lines.append('ANNOTATION 3: ')
            lines.append('END_EXAMPLE')
            lines.append('\n\n')
        with open(annotation_fn, 'w') as fd:
            fd.write('\n'.join(lines))
        exit(0)

    hf = 'microsoft/deberta-large-mnli'
    bs = BERTScorer(model_type=hf)
    model = AutoModel.from_pretrained(hf).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(hf)

    annotation_fn = os.path.expanduser(f'~/faith-sum/preprocess/analysis/{args.dataset}_alignment_annotations.txt')
    nlp = spacy.load('en_core_web_sm')
    examples, annots_len = read_annotations(annotation_fn, nlp)

    print(f'Average annotation length in sentences: ', np.mean(annots_len))

    if args.algorithm == 'rouge':
        # Benchmark against ROUGE Gain
        rouge_outputs = []
        lens, ps, rs, f1s = [], [], [], []
        bs_align = []
        for example in tqdm(examples):
            annotations = example['annotations']
            source_sents_tok = [[str(token.text) for token in nlp(sentence)] for sentence in example['dialogue']]
            target_sents_tok = [[str(token.text) for token in nlp(sentence)] for sentence in example['summary']]
            # Sort oracle order or not
            align_idxs, rouge, r1_hist, r2_hist, best_hist = gain_selection(
                source_sents_tok, target_sents_tok, 5, lower=True, sort=False
            )
            rouge_outputs.append(align_idxs)
            p, r, f1 = compute_alignment_score(align_idxs, annotations)
            bs_align.append(
                bs_overlap(bs, align_idxs, annotations, example['dialogue'])
            )

            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            lens.append(len(align_idxs))
        print('Rouge gain alignments (BS F1 / P / R / F1 / # Sent)...')
        print(np.mean(bs_align), np.mean(ps), np.mean(rs), np.mean(f1s), np.mean(lens))
        exit(0)

    if args.mode == 'tune':
        if args.algorithm == 'bert_coverage':
            if args.dataset == 'cnn_dailymail':
                avg_thresholds = [0.1, 0.15, 0.2]
                max_imp_thresholds = [0.6, 0.7, 0.8]
                max_retrieval_candidates = [2, 3, 4]
                max_coverages = [0.8, 0.9, 1.0]
            elif args.dataset == 'xsum':
                avg_thresholds = [0.05, 0.075, 0.1]
                max_imp_thresholds = [0.1, 0.15, 0.2]
                max_retrieval_candidates = [7, 8, 9]
                max_coverages = [0.85, 0.9, 0.95, 1.0]
            else:
                avg_thresholds = [0.01, 0.02, 0.03]
                max_imp_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
                max_retrieval_candidates = [3, 4, 5]
                max_coverages = [0.8, 0.9, 0.95, 1.0]
            tune_coverage(
                args, model, tokenizer, bs, avg_thresholds, max_imp_thresholds, max_coverages, max_retrieval_candidates
            )
        elif args.algorithm == 'bert_align':
            thresholds = [0.55, 0.560, 0.565, 0.57, 0.575]
            max_retrieval_candidates = [2, 3, 4]
            p_factors = [1.1, 1.2, 1.3, 1.4]
            tune_align(args, bs, thresholds, max_retrieval_candidates, p_factors)
        else:
            raise Exception(f'Unrecognized algorithm --> {args.algorithm}')
        exit(0)

    lens, ps, rs, f1s = [], [], [], []
    new_lens, new_ps, new_rs, new_f1s = [], [], [], []
    new_bs = []
    bert_outputs = []

    # Best
    avg_imp_threshold = 0.02
    max_imp_threshold = 0.15
    max_coverage = 0.95
    max_retrievals = 4

    output_str = []

    for example in tqdm(examples):
        annotations = example['annotations']
        new_align_idxs, new_align_sents = add_bert_alignment_no_red(
            model, tokenizer, example['dialogue'], example['summary'],
            avg_imp_threshold=avg_imp_threshold,
            max_imp_threshold=max_imp_threshold,
            max_retrievals=max_retrievals,
            max_coverage=max_coverage
        )

        new_bs.append(bs_overlap(bs, new_align_idxs, annotations, example['dialogue']))
        new_p, new_r, new_f1 = compute_alignment_score(new_align_idxs, annotations)
        new_ps.append(new_p)
        new_rs.append(new_r)
        new_f1s.append(new_f1)
        new_lens.append(len(new_align_idxs))
        output_str.append(','.join([str(x) for x in list(sorted(new_align_idxs))]))

        # align_idxs, align_sents = add_bert_alignment(
        #     bs, example['dialogue'], example['summary'], threshold=threshold, p_factor=p_factor, max_per_sent=max_per_sent
        # )
        # bert_outputs.append(align_idxs)
        # p, r, f1 = compute_alignment_score(align_idxs, example['annotations'])
        # ps.append(p)
        # rs.append(r)
        # f1s.append(f1)
        # lens.append(len(align_idxs))

    print({
        'p': float(np.mean(new_ps)),
        'r': float(np.mean(new_rs)),
        'f1': float(np.mean(new_f1s)),
        'bs_align': float(np.mean(new_bs)),
        'num_sents': float(np.mean(new_lens)),
    })
