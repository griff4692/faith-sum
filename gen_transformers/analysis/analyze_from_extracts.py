import os
import itertools

from scipy.stats import pearsonr
import pandas as pd
import regex as re
from p_tqdm import p_uimap
from datasets import load_from_disk
from preprocess.convert_abstractive_to_extractive import gain_selection
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.align_edu import edus_from_html


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


def process(record, nlp, source_annotated):
    if '<cand>' in record['extract_idx']:
        extract_idxs = record['extract_idx'].split('<cand>')
        extracts = record['extract'].split('<cand>')
        abstracts = record['from_extract_abstract'].split('<cand>')
        from_extract_rouges = get_arr(record['eval_from_extract_abstract_rouge1_f1'])
        extract_rouges = get_arr(record['eval_extract_rouge1_f1'])
    else:
        extract_idxs = [record['extract_idx']]
        abstracts = [record['from_extract_abstract']]
        extracts = [record['extract']]
        from_extract_rouges = [record['eval_from_extract_abstract_rouge1_f1']]
        extract_rouges = [record['eval_extract_rouge1_f1']]

    rows = []
    beam = -1
    for extract_idx, extract, abstract, extract_rouge, from_extract_rouge in zip(
            extract_idxs, extracts, abstracts, extract_rouges, from_extract_rouges
    ):
        beam += 1
        extract_idx = list(map(int, get_arr(extract_idx)))
        source_sents = edus_from_html(source_annotated[record['dataset_idx']])

        source_sents_tok = [[str(token.text) for token in nlp(sentence)] for sentence in source_sents]
        target_sents = [x for x in list(nlp(abstract.strip()).sents) if len(x.text.strip()) > 0]
        target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]

        extract_toks = set([str(token.text) for token in nlp(extract)])
        target_toks = set(list(itertools.chain(*target_sents_tok)))

        compression = len(extract_toks) / len(target_toks)

        tok_agreement = extract_toks.intersection(target_toks)
        tok_coverage = len(tok_agreement) / len(target_toks)

        implied_extract_obj = gain_selection(source_sents_tok, target_sents_tok, 20, lower=True, sort=True)

        implied_idx = implied_extract_obj[0]
        agreement = set(extract_idx).intersection(implied_idx)

        n = len(agreement)
        r = n / len(extract_idx)
        p = n / len(implied_idx)
        f1 = 0 if min(r, p) == 0 else (2 * p * r) / (p + r)

        row = {
            'num_abstract_sents': len(target_sents_tok),
            'num_implied_sents': len(implied_idx),
            'num_extract_sents': len(extract_idx),
            'plan_recall': r,
            'plan_precision': p,
            'plan_f1': f1,
            'plan_coverage': tok_coverage,
            'rouge_advantage': from_extract_rouge - extract_rouge,
            'beam': beam,
            'compression': compression,
        }
        rows.append(row)
    return rows


if __name__ == '__main__':
    dataset = 'cnn_dailymail'
    data_dir = '/nlp/projects/faithsum'
    experiment = 'add_doc_bart_large_cnn'
    output = 'test_from_diverse_16_extract'
    summary_style = 'from_extract_abstract'



    df = pd.read_csv(f'{data_dir}/results/{experiment}/{output}.csv').dropna(subset=[summary_style])

    records = df.to_dict('records')

    nlp = spacy.load('en_core_web_sm')
    orig_data_dir = os.path.join(data_dir, dataset)
    print('Loading original data')
    dataset = load_from_disk(orig_data_dir)['test']
    source_annotated = dataset['source_annotated']

    stats = list(itertools.chain(*list(p_uimap(lambda record: process(record, nlp, source_annotated), records))))
    stats = pd.DataFrame(stats)
    for col in stats.columns:
        print(col, stats[col].dropna().mean())
        if col != 'rouge_advantage':
            print('\t- ', col, pearsonr(stats[col], stats['rouge_advantage'])[0])

    print(pearsonr(stats['num_extract_sents'], stats['plan_f1'])[0])
    print(pearsonr(stats['beam'], stats['rouge_advantage'])[0])

    # Remove example_id
    corel_cols = list(sorted([x for x in stats.select_dtypes('number').columns.tolist() if 'id' not in x]))
    corel_fn = 'correlations.png'
    corel_df = stats[corel_cols]
    fig, ax = plt.subplots(figsize=(8, 8))
    # sns.set(font_scale=2)
    sns.heatmap(corel_df.corr(), annot=True, linewidths=0.5, ax=ax, fmt='.2f', cmap='coolwarm')
    plt.tight_layout()
    print(f'Saving correlation matrix for {len(corel_cols)} variables to {corel_fn}')
    plt.savefig(corel_fn, bbox_inches='tight')
