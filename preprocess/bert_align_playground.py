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


def encode(text, model, tokenizer, device, max_length=128, max_batch_size=100, top_n_layers=1):
    seq_lens = []
    outputs = {'hidden_states': [], 'logits': []}
    text_batches = [list(x) for x in np.array_split(np.arange(len(text)), round(len(text) // max_batch_size) + 1)]
    text_batches = [x for x in text_batches if len(x) > 0]
    inputs = tokenizer(text, truncation=True, padding='longest', max_length=max_length, return_tensors='pt')

    stopword_masks = []

    # for idx, id in enumerate(inputs['input_ids'].cpu().numpy()):
    #     seq_len = int(inputs['attention_mask'][idx].sum().item())
    #     toks = tokenizer.convert_ids_to_tokens(id)
    #     is_stop = [x.lower().strip('Ġ') in STOPWORDS or x.lower().strip('Ġ') in string.punctuation for x in toks]
    #     is_stop = is_stop[:seq_len]
    #     is_stop[0] = True
    #     is_stop[-1] = True
    #     stopword_masks.append(is_stop)

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


# def add_bert_alignment(nlp, batch_data, target_col, bs):
#     target = batch_data[target_col]
#     target_sents = list(map(str, convert_to_sents(target, nlp, is_dialogue=False)))
#     source_sents = []
#     tps = re.split(r'(<s\d+>)', batch_data['source_annotated'])
#     for tp_idx, tp in enumerate(tps):
#         if re.match(r'(<s\d+>)', tp) is not None:
#             source_sents.append(tps[tp_idx + 1].strip())
#
#     bert_idxs, bert_alignments = _add_bert_alignment(bs, source_sents, target_sents)
#
#     rouge_oracle = list(sorted(batch_data['oracle_idxs']))
#     if bert_idxs == rouge_oracle:
#         print('equal')
#     else:
#         print('\n', bert_idxs, rouge_oracle)
#         print(bert_alignments)
#         print('\n')
#         print(batch_data['source_annotated'])
#         print('\n')
#         print('Summary: ', target)
#         print('\n\n\n')
#         print('not equal')
#
#     return {
#         'bert_idxs': bert_idxs,
#         'bert_alignments': bert_alignments
#     }


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--splits', default='validation')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('-save_annotation_examples', default=False, action='store_true')

    args = parser.parse_args()

    hf = 'microsoft/deberta-large-mnli'
    model = AutoModel.from_pretrained(hf).eval().to(0)
    tokenizer = AutoTokenizer.from_pretrained(hf)

    if args.save_annotation_examples:
        input_col, target_col = summarization_name_mapping[args.dataset]
        out_dir = os.path.join(args.data_dir, args.dataset)
        dataset = load_from_disk(out_dir)
        print('Save annotation examples')
        val = dataset['validation']
        rand_idxs = list(np.sort(np.random.choice(np.arange(len(val)), size=(30, ), replace=False)))
        val_subset = val.select(rand_idxs)
        annotation_fn = os.path.expanduser('~/samsum_annotations.txt')
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

    annotation_fn = os.path.expanduser('~/faith-sum/preprocess/analysis/samsum_alignment_annotations.txt')
    nlp = spacy.load('en_core_web_sm')
    annots_len = []

    with open(annotation_fn, 'r') as fd:
        lines = fd.readlines()
        lines = [x.strip() for x in lines if len(x.strip()) > 0]

        examples = []
        for idx, line in enumerate(lines):
            if line == 'BEGIN_EXAMPLE':
                curr_dialogue = []
                curr_annotations = []
                curr_summary = ''
            elif line.startswith('<s'):
                curr_dialogue.append(re.sub('<s\d+> ', '', line))
            elif line.startswith('ANNOTATION'):
                annots = [int(x.strip()) for x in line.split(':')[1].strip().split(',') if len(x.strip()) > 0]
                if len(annots) > 0:
                    annots_len.append(len(annots))
                    curr_annotations.append(annots)
            elif line == 'SUMMARY:':
                curr_summary = [str(x) for x in list(nlp(lines[idx + 1]).sents) if len(x) > 0]
            elif line == 'END_EXAMPLE':
                examples.append({
                    'dialogue': curr_dialogue,
                    'annotations': curr_annotations,
                    'summary': curr_summary,
                })

    print(f'Average annotation length in sentences: ', np.mean(annots_len))

    # Benchmark against ROUGE Gain
    # rouge_outputs = []
    # lens, ps, rs, f1s = [], [], [], []
    # for example in tqdm(examples):
    #     source_sents_tok = [[str(token.text) for token in nlp(sentence)] for sentence in example['dialogue']]
    #     target_sents_tok = [[str(token.text) for token in nlp(sentence)] for sentence in example['summary']]
    #     # Sort oracle order or not
    #     align_idxs, rouge, r1_hist, r2_hist, best_hist = gain_selection(
    #         source_sents_tok, target_sents_tok, 5, lower=True, sort=False
    #     )
    #     rouge_outputs.append(align_idxs)
    #     p, r, f1 = compute_alignment_score(align_idxs, example['annotations'])
    #     ps.append(p)
    #     rs.append(r)
    #     f1s.append(f1)
    #     lens.append(len(align_idxs))
    #
    # print('Rouge gain alignments (P / R / F1 / # Sent)...')
    # print(np.mean(ps), np.mean(rs), np.mean(f1s), np.mean(lens))

    # avg_imp_threshold = 0.01, max_imp_threshold = 0.05, max_retrievals = 3, max_coverage = 0.85,
    # avg_thresholds = [0.01, 0.05, 0.02, 0.025]
    # max_imp_thresholds = [0.05, 0.1, 0.15]
    # max_retrieval_candidates = [2, 3, 4, 5, 6, 7]
    # max_coverages = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    #
    # outputs = []
    # best_f1 = 0
    # for avg_imp_threshold in avg_thresholds:
    #     avg_imp_threshold = float(avg_imp_threshold)
    #     for max_imp_threshold in max_imp_thresholds:
    #         max_imp_threshold = float(max_imp_threshold)
    #         for max_coverage in max_coverages:
    #             for max_retrievals in max_retrieval_candidates:
    #                 ps, rs, f1s = [], [], []
    #                 for example in tqdm(examples):
    #                     align_idxs, _ = add_bert_alignment_no_red(
    #                         model, tokenizer, example['dialogue'], example['summary'],
    #                         avg_imp_threshold=avg_imp_threshold,
    #                         max_imp_threshold=max_imp_threshold,
    #                         max_coverage=max_coverage,
    #                         max_retrievals=max_retrievals
    #                     )
    #
    #                     p, r, f1 = compute_alignment_score(align_idxs, example['annotations'])
    #                     ps.append(p)
    #                     rs.append(r)
    #                     f1s.append(f1)
    #
    #                 old_best = best_f1
    #                 avg_f1 = float(np.mean(f1s))
    #                 best_f1 = max(avg_f1, best_f1)
    #                 if old_best != best_f1:
    #                     print('\n', best_f1)
    #
    #                 outputs.append({
    #                     'avg_imp_threshold': avg_imp_threshold,
    #                     'avg_max_threshold': max_imp_threshold,
    #                     'max_coverage': max_coverage,
    #                     'max_retrievals': max_retrievals,
    #                     'p': float(np.mean(ps)),
    #                     'r': float(np.mean(rs)),
    #                     'f1': avg_f1,
    #                 })
    # outputs = pd.DataFrame(outputs)
    # outputs = outputs.sort_values(by='f1', ascending=False).reset_index(drop=True)
    # print(outputs.head(n=10))
    # outputs.to_csv('tmp.csv', index=False)
    # exit(0)

    # bs = BERTScorer(model_type='microsoft/deberta-large-mnli')  #  lang='en')
    # threshold = 0.87
    # p_factor = 0.8
    # max_per_sent = 3

    lens, ps, rs, f1s = [], [], [], []
    new_lens, new_ps, new_rs, new_f1s = [], [], [], []
    bert_outputs = []

    # Best
    avg_imp_threshold = 0.02
    max_imp_threshold = 0.15
    max_coverage = 0.95
    max_retrievals = 4

    output_str = []

    for example in tqdm(examples):
        new_align_idxs, new_align_sents = add_bert_alignment_no_red(
            model, tokenizer, example['dialogue'], example['summary'],
            avg_imp_threshold=avg_imp_threshold,
            max_imp_threshold=max_imp_threshold,
            max_retrievals=max_retrievals,
            max_coverage=max_coverage
        )

        new_p, new_r, new_f1 = compute_alignment_score(new_align_idxs, example['annotations'])
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
        'num_sents': float(np.mean(new_lens)),
    })

    # print('\n'.join([f'BERT (Red):{x}' for x in output_str]))
    # print({
    #     'threshold': threshold,
    #     'p_factor': p_factor,
    #     'p': float(np.mean(ps)),
    #     'r': float(np.mean(rs)),
    #     'f1': float(np.mean(f1s)),
    #     'num_sents': float(np.mean(lens)),
    # })

    #
    # for example_idx, example in enumerate(examples):
    #     dialogue_str = '\n'.join([f'<s{d}> {t}' for d, t in enumerate(example['dialogue'])])
    #     summary_str = ' '.join(example['summary'])
    #     rouge = list(sorted(rouge_outputs[example_idx]))
    #     bert = list(sorted(bert_outputs[example_idx]))
    #
    #     if rouge != bert:
    #         print('DIALOGUE:')
    #         print(dialogue_str)
    #         print('\n')
    #         print('SUMMARY:')
    #         print(summary_str)
    #         print('\n')
    #         print('ROUGE: ', rouge)
    #         print('HUMAN: ', list(sorted(example['annotations'])))
    #         print('BERT : ', bert)
    #         print('\n\n')

    # thresholds = np.arange(0.5, 0.725, 0.025).tolist()
    # p_factors = np.arange(0.75, 1.255, 0.05).tolist()
    # outputs = []
    # best_f1 = 0
    # for p_factor in p_factors:
    #     p_factor = float(p_factor)
    #     for threshold in thresholds:
    #         threshold = float(threshold)
    #         ps, rs, f1s = [], [], []
    #         for example in tqdm(examples):
    #             align_idxs, align_sents = add_bert_alignment(
    #                 bs, example['dialogue'], example['summary'], threshold=threshold, p_factor=p_factor
    #             )
    #
    #             p, r, f1 = compute_alignment_score(align_idxs, example['annotations'])
    #             ps.append(p)
    #             rs.append(r)
    #             f1s.append(f1)
    #
    #         old_best = best_f1
    #         avg_f1 = float(np.mean(f1s))
    #         best_f1 = max(avg_f1, best_f1)
    #         if old_best != best_f1:
    #             print('\n', best_f1)
    #
    #         outputs.append({
    #             'threshold': threshold,
    #             'p_factor': p_factor,
    #             'p': float(np.mean(ps)),
    #             'r': float(np.mean(rs)),
    #             'f1': float(np.mean(f1s)),
    #         })
    # outputs = pd.DataFrame(outputs)
    # outputs = outputs.sort_values(by='f1', ascending=False).reset_index(drop=True)
    # print(outputs.head(n=10))
    # outputs.to_csv('tmp_bert.csv', index=False)
