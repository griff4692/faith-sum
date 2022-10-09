from datasets import load_from_disk
import regex as re


x = load_from_disk('/nlp/projects/faithsum/cnn_dailymail')

val = x['validation']

sources = val['source_annotated']
highlights = val['highlights']
oracle_rouges = val['oracle_rouge1']
oracle_idxs = val['oracle_idxs']

import numpy as np

oracle_rouge_order = np.argsort(oracle_rouges)
k = 10

best_oracles = oracle_rouge_order[-k:]
n = len(oracle_rouge_order)
mid = n // 2
mid_oracles = oracle_rouge_order[mid - k // 2:mid + k // 2]
worst_oracles = oracle_rouge_order[:k]


def render(idx):
    lines = []
    extract_idxs = oracle_idxs[idx]
    extract_idx_str = ', '.join([str(x) for x in extract_idxs])
    source = sources[idx]
    ref = highlights[idx].strip()

    tps = re.split(r'(<s\d+>)', source)
    source_sents = []
    for tp_idx, tp in enumerate(tps):
        if re.match(r'(<s\d+>)', tp) is not None:
            source_sents.append(tps[tp_idx + 1].strip())

    extract_sents = []
    for extract_idx in np.sort(extract_idxs):
        extract_sents.append(source_sents[extract_idx])

    extract = ' '.join(extract_sents)
    lines.append(f'SOURCE: {source}\n')
    lines.append(f'REFERENCE: {ref}\n')
    lines.append(f'ORACLE ROUGE: {oracle_rouges[idx]}\n')
    lines.append(f'ORACLE INDICES: {extract_idx_str}\n')
    lines.append(f'ORACLE: {extract}')
    return '\n'.join(lines)


sample_idxs = np.random.choice(np.arange(n), size=(25, ), replace=False)

all_lines = []
for idx in sample_idxs:
    all_lines.append(render(idx))
    all_lines.append('\n')
    all_lines.append('-' * 50)
    all_lines.append('\n')

with open('/home/griffin/cnn_oracles.txt', 'w') as fd:
    fd.writelines(all_lines)


# print('\nBest Oracles...\n')
# for idx in best_oracles:
#     render(idx)
#
# print('\n\n\n\nWorst Oracles...\n')
# for idx in worst_oracles:
#     render(idx)
#
# print('\n\n\n\nMid Oracles...\n')
# for idx in mid_oracles:
#     render(idx)
