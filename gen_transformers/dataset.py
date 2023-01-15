import ujson
import os
from string import punctuation

import numpy as np
from nltk.corpus import stopwords
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
STOPWORDS = set(stopwords.words('english'))
from datasets import load_from_disk

from gen_transformers.data_utils import Seq2SeqCollate
from preprocess.helpers import _get_ngrams
from sum_constants import summarization_name_mapping
from preprocess.align_edu import edus_from_html


BRIO_EXPS = {
    'samsum': '/nlp/projects/faithsum/results/samsum_bert_red_extract_generator_3e5lr/{}_from_beam_10_extract.csv'
}


def remove_stopwords(tokens):
    return [w for w in tokens if not w.lower() in STOPWORDS and w not in punctuation and len(w.strip()) > 0]


def remove_non_oracle(input_ids, oracle_labels, special_token_ids):
    s, e = special_token_ids
    start_locs = np.where(np.array(input_ids) == s)[0]
    end_locs = np.where(np.array(input_ids) == e)[0]

    n = len(start_locs)
    remove_idxs = []
    for idx in range(n):
        if idx not in oracle_labels:
            remove_idxs += [start_locs[idx], end_locs[idx]]

    keep_idxs = np.sort(list(set(list(range(len(input_ids)))) - set(remove_idxs)))
    return [input_ids[i] for i in keep_idxs]


def corrupt_indicators(input_ids, oracle_idxs, special_token_ids, corrupt_strategy):
    s, e = special_token_ids
    start_locs = np.where(np.array(input_ids) == s)[0]
    end_locs = np.where(np.array(input_ids) == e)[0]

    n = len(start_locs)
    oracle_n = len(oracle_idxs)
    assert n == len(end_locs)

    non_oracle_idxs = [i for i in range(n) if i not in oracle_idxs]
    non_oracle_n = len(non_oracle_idxs)

    if corrupt_strategy == 'random':
        num_to_replace = min(non_oracle_n, oracle_n)
        idx_to_keep = np.sort(np.random.choice(non_oracle_idxs, size=(num_to_replace,), replace=False))
    else:
        assert corrupt_strategy == 'swap'
        idx_to_keep = oracle_idxs.copy()
        if non_oracle_n >= 1:
            other_sent = int(np.random.choice(non_oracle_idxs))
            idx_to_keep[np.random.randint(oracle_n)] = other_sent
            idx_to_keep = list(np.sort(idx_to_keep))
        else:
            idx_to_keep = idx_to_keep[:-1]
    return remove_non_oracle(input_ids, idx_to_keep, special_token_ids)


def get_sent_ngrams(source_annotated):
    source_edus = edus_from_html(source_annotated)
    def get_ngrams(edu):
        toks = list(map(lambda x: x.lower(), edu.split('\W+')))
        return [_get_ngrams(1, toks), _get_ngrams(2, toks), _get_ngrams(3, toks)]
    source_ngrams = list(map(get_ngrams, source_edus))
    return source_ngrams


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        pegasus_suffix = '_pegasus' if 'pegasus' in args.hf_model else ''
        data_dir = os.path.join(args.data_dir, args.dataset + f'_edu_alignments{pegasus_suffix}')
        print(f'Loading data from {data_dir}')
        self.dataset = load_from_disk(data_dir)
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 8
        self.nlp = spacy.load('en_core_web_sm')

    def get_train_chunk(self, chunk, num_chunks, **dataloader_kwargs):
        split_dataset = self.dataset['train']
        n = len(split_dataset)

        all_idxs = list(range(n))
        chunk_idxs = np.array_split(all_idxs, num_chunks)[chunk]
        print(f'Using {len(chunk_idxs)} training examples set for chunk {chunk}/{num_chunks}')
        print(f'First {min(3, len(chunk_idxs))} idxs: {chunk_idxs[:min(3, len(chunk_idxs))]}')
        split_dataset = split_dataset.select(chunk_idxs)

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, self.tokenizer, 'train')
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            split='train',
        )
        kwargs = {
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        kwargs.update(**dataloader_kwargs)
        return DataLoader(split_dataset_pl, **kwargs), chunk_idxs

    def get_split(self, split, max_examples=None, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        if self.args.debug and max_examples is None:
            max_examples = 128

        brio_candidates = None
        if self.args.add_brio_loss:
            if self.args.use_oracle_candidates:
                out_dir = os.path.join(self.args.data_dir, self.args.dataset, 'oracle')
                out_fn = os.path.join(out_dir, f'{split}_candidates.json')
                with open(out_fn, 'r') as fd:
                    candidates = ujson.load(fd)

                brio_candidates = {}
                for dataset_id, cands in candidates.items():
                    scores = [float(x) for x in cands['ea']['from_extract_rouges'].split('<cand>')]
                    extract_idxs = [x['extract_idx'] for x in cands['oracles']]
                    order = np.argsort(-np.array(scores))
                    scores_ordered = [scores[i] for i in order]
                    extract_idxs_ordered = [extract_idxs[i] for i in order]
                    # scores_ordered = [x['mean_f1'] for x in cands]
                    # extract_idxs_ordered = [x['extract_idx'] for x in cands]
                    for i in range(1, len(scores_ordered)):  # Assert it's pre-sorted by ROUGE
                        assert scores_ordered[i - 1] >= scores_ordered[i]
                    if len(cands) < 2:
                        continue
                    brio_candidates[dataset_id] = [extract_idxs_ordered, scores_ordered]
            else:
                predictions_df = pd.read_csv(BRIO_EXPS[self.args.dataset].format(split))
                brio_candidates = {}
                dataset_ids = split_dataset['id']
                for record in predictions_df.to_dict('records'):
                    extracts = [[int(y) for y in cand.split(',')] for cand in record['extract_idx'].split('<cand>')]
                    ea_rouges = [
                        float(x) for x in record['from_extract_rouges'].split('<cand>')
                    ]

                    order = np.argsort(-np.array(ea_rouges))
                    extract_idxs_ordered = [extracts[i] for i in order]
                    rouges_ordered = [ea_rouges[i] for i in order]
                    if len(extract_idxs_ordered) < 2:
                        continue
                    brio_candidates[dataset_ids[record['dataset_idx']]] = [extract_idxs_ordered, rouges_ordered]

            # Filter dataset to only include ones with BRIO candidates generated or 'oracled'
            available_keys = set(list(brio_candidates.keys()))
            valid_hf_ids = [i for i, dataset_idx in enumerate(split_dataset['id']) if dataset_idx in available_keys]
            print(f'Filtering out for {len(valid_hf_ids)}/{len(split_dataset)} contrastive BRIO examples')
            split_dataset = split_dataset.select(valid_hf_ids)

        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
            split_dataset = split_dataset.select(idxs)

        # if split == 'train' and self.args.oracle_drop_p > 0:
        #     oracle_rouge_1 = split_dataset['oracle_rouge1']
        #     oracle_rouge_2 = split_dataset['oracle_rouge2']
        #     avg_rouge = [(a + b) / 2.0 for (a, b) in zip(oracle_rouge_1, oracle_rouge_2)]
        #     priority = np.argsort(avg_rouge)
        #     drop_n = round(len(split_dataset) * self.args.oracle_drop_p)
        #     print(f'Filtering out {drop_n} training examples with lowest oracle score.')
        #     keep_idxs = list(sorted(priority[drop_n:]))
        #     split_dataset = split_dataset.select(keep_idxs)

        split_dataset_pl = SummarizationDataset(
            self.args, split_dataset, self.tokenizer, split, brio_candidates=brio_candidates
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            split=split,
        )
        batch_size = self.args.per_device_train_bs if split == 'train' else self.args.per_device_eval_bs
        kwargs = {
            'batch_size': batch_size,
            'shuffle': split == 'train',
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        kwargs.update(**dataloader_kwargs)
        return DataLoader(split_dataset_pl, **kwargs), idxs

    def train_dataloader(self, max_examples=None):
        return self.get_split('train', max_examples=None)[0]

    def val_dataloader(self, max_examples=None):
        return self.get_split('validation', max_examples=max_examples or self.args.max_val_examples)[0]

    def test_dataloader(self, max_examples=None):
        return self.get_split('test', max_examples=max_examples)[0]


class SummarizationDataset(Dataset):
    def __init__(self, args, dataset, tokenizer, split, brio_candidates=None):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.split = split
        self.brio_candidates = brio_candidates
        self.input_col, self.target_col = summarization_name_mapping[self.args.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        dataset_id = example['id']
        target = example[self.target_col]

        oracle_labels = np.sort(example['oracle_idxs'])
        oracle_soft_labels = example['oracle_soft_labels']
        assert len(example['oracle_soft_labels']) == len(
            [x for x in example['input_ids'] if x == self.tokenizer.additional_special_tokens_ids[0]]
        )
        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        source_annotated = example['source_edu_annotated']
        input_ids = example['input_ids']
        corrupt_input_ids = None
        plan_input_ids = None
        if not self.args.add_sent_toks:
            input_ids = [x for x in input_ids if x not in self.tokenizer.additional_special_tokens_ids]
        elif self.args.extract_indicators:
            input_ids = [x for x in input_ids if x not in self.tokenizer.additional_special_tokens_ids]
            # Remove Non-Oracle Markers
            corrupt_input_ids = corrupt_indicators(
                input_ids.copy(), oracle_labels.copy(), self.tokenizer.additional_special_tokens_ids,
                self.args.corrupt_strategy
            )
            plan_input_ids = remove_non_oracle(input_ids, oracle_labels, self.tokenizer.additional_special_tokens_ids)

        row = {
            'input_ids': input_ids,
            'labels': example['labels'],
            'source': source_annotated,
            'oracle_labels': oracle_labels,
            'oracle_soft_labels': oracle_soft_labels,
            'reference': target,  # Use for evaluation
            'source_ngrams': get_sent_ngrams(source_annotated)
        }

        if corrupt_input_ids is not None:
            row['corrupt_input_ids'] = corrupt_input_ids
            row['plan_input_ids'] = plan_input_ids

        if self.args.debug:
            source_edus = edus_from_html(source_annotated)
            extract = [source_edus[i] for i in oracle_labels]
            print(extract)
            print(target)

        if self.args.add_brio_loss:
            candidates, scores = self.brio_candidates[dataset_id].copy()  # We modify it in place so let's insert

            if len(candidates) > self.args.max_brio_candidates:
                candidates = candidates[:self.args.max_brio_candidates]
                scores = scores[:self.args.max_brio_candidates]

            scores = np.array(scores)
            norm_scores = (scores - min(scores)) / (max(scores) - min(scores))

            oracle_in_list = any([
                list(oracle_labels) == cand for cand in candidates
            ])
            # If we want to include the oracle in the Gold
            if not oracle_in_list and self.args.include_gold:  # If the model didn't already generate the oracle
                # Add Gold Label as the 'most positive'
                candidates.insert(0, list(oracle_labels))
                norm_scores.insert(0, 1)

            row['brio_sent_labels'] = candidates
            row['brio_norm_scores'] = norm_scores
        return row
