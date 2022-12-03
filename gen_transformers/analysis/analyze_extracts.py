import os
import pandas as pd
from collections import Counter
from datasets import load_from_disk
from scipy.stats import pearsonr
import numpy as np


DIR = '/nlp/projects/faithsum/results'


def get_num_extracts(df):
    nums = []
    for cands in df['extract_idx'].tolist():
        for cand in cands.split('<cand>'):
            nums.append(len(cand.split(',')))
    return Counter(nums)


def extract_abstract_corel(df):
    corels = []

    for record in df.to_dict('records'):
        extract_len = []
        abstract_toks = []
        for cand in record['extract_idx'].split('<cand>'):
            extract_len.append(len(cand.split(',')))
        for cand in record['from_extract_abstract'].split('<cand>'):
            abstract_toks.append(len(cand.split(' ')))

        if min(abstract_toks) == max(abstract_toks):
            continue

        if min(extract_len) == max(extract_len):
            continue
        corels.append(float(pearsonr(abstract_toks, extract_len)[0]))
    print(np.mean(corels))


if __name__ == '__main__':
    fn1 = os.path.join(DIR, 'samsum_bert_red_extract_generator_3e5lr', 'validation_from_beam_16_extract.csv')
    bert = pd.read_csv(fn1)

    fn2 = os.path.join(DIR, 'samsum_extract_generator', 'validation_from_beam_16_extract.csv')
    rouge = pd.read_csv(fn2)

    print(extract_abstract_corel(bert))
    print(extract_abstract_corel(rouge))

    val = load_from_disk('/nlp/projects/faithsum/samsum')['validation']

    for i in range(len(bert)):
        outputs = list(zip(bert.extract_idx[i].split('<cand>'), bert.from_extract_abstract[i].split('<cand>')))
        sa = val[i]['source_annotated']
        print(sa)
        print('\n')
        for e, ea in outputs:
            print(e + ' -> ' + ea)

        print('\n')
        print('-' * 100)
        print('\n')
        if i > 10:
            break

    exit(0)

    bert_lens = get_num_extracts(bert)
    rouge_lens = get_num_extracts(rouge)
    print('BERT Lengths')
    for i in range(1, 11):
        print(bert_lens.get(i, 0) / sum(bert_lens.values()))

    print('ROUGE Lengths')
    for i in range(1, 11):
        print(rouge_lens.get(i, 0) / sum(rouge_lens.values()))

    rouge_oracle_lens = Counter([len(x) for x in val['oracle_idxs']])
    bert_oracle_lens = Counter([len(x) for x in val['oracle_idxs_bert']])

    print('BERT ORACLE Lengths')
    for i in range(1, 11):
        print(bert_oracle_lens.get(i, 0) / sum(bert_oracle_lens.values()))

    print('ROUGE ORACLE Lengths')
    for i in range(1, 11):
        print(rouge_oracle_lens.get(i, 0) / sum(rouge_oracle_lens.values()))
