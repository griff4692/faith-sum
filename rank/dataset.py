from collections import defaultdict
import os
import itertools

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import spacy

from sum_constants import summarization_name_mapping


class RankCollate:
    def __init__(self, tokenizer, max_input_length=8192, max_output_length=512, add_cols=None, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.add_cols = [] if add_cols is None else add_cols
        self.split = split

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            [x['source'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        flat_targets = list(itertools.chain(*[x['targets'] for x in batch_list]))

        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(
                flat_targets,
                padding='longest',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors='pt'
            )

        batch = {}
        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        batch['labels'] = outputs.input_ids
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100

        for col in self.add_cols:
            batch[col] = [x[col] for x in batch_list]

        return batch


class RankDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 16
        self.nlp = spacy.load('en_core_web_sm')

        beam_fn = os.path.join(self.args.data_dir, 'results', self.args.gen_experiment, 'validation_beam_outputs.csv')
        sample_fn = os.path.join(
            self.args.data_dir, 'results', self.args.gen_experiment, 'validation_sample_outputs.csv'
        )

        beam_df = pd.read_csv(beam_fn)
        sample_df = pd.read_csv(sample_fn)

        avail_idxs = sample_df['dataset_idx'].unique().tolist()
        beam_df = beam_df[beam_df['dataset_idx'].isin(set(avail_idxs))]

        sample_records = {record['dataset_idx']: record for record in sample_df.to_dict('records')}
        beam_records = {record['dataset_idx']: record for record in beam_df.to_dict('records')}
        np.random.seed(1992)
        train_frac = 0.75
        n = len(sample_records)
        train_cutoff = round(train_frac * n)
        np.random.shuffle(avail_idxs)

        self.splits = {
            'train': avail_idxs[:train_cutoff],
            'validation': avail_idxs[train_cutoff:]
        }

        num_train = len(self.splits['train'])
        num_val = len(self.splits['validation'])
        print(f'{num_train} train examples. {num_val} validation.')

        combined_data = defaultdict(dict)
        for dataset_idx in avail_idxs:
            combined_data[dataset_idx] = {
                'dataset_idx': dataset_idx,
                'beam': beam_records[dataset_idx],
                'sample': sample_records[dataset_idx]
            }
        self.dataset = list(combined_data.values())

    def get_split(self, split, max_examples=None):
        split_dataset = list(filter(lambda example: example['dataset_idx'] in self.splits[split], self.dataset))
        if self.args.debug and max_examples is None:
            max_examples = 128
        n = len(split_dataset)
        rand_idxs = None
        if max_examples is not None and max_examples < n:
            rand_idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = [split_dataset[i] for i in rand_idxs]
        add_cols = ['extract_scores', 'abstract_scores', 'avg_scores']
        split_dataset_pl = RankDataset(self.args, split_dataset, split, self.nlp, add_cols=add_cols)
        collate_fn = RankCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            add_cols=add_cols,
            split=split
        )
        kwargs = {
            'batch_size': 1,
            'shuffle': split == 'train',
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(split_dataset_pl, **kwargs), rand_idxs

    def train_dataloader(self, max_examples=None):
        return self.get_split('train', max_examples=None)[0]

    def val_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('validation', max_examples=max_examples or self.args.max_val_examples)[0]

    def test_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('test', max_examples=max_examples)[0]


class RankDataset(Dataset):
    def __init__(self, args, dataset, split, nlp, add_cols=None, ids2oracles=None):
        super(RankDataset, self).__init__()
        self.args = args
        self.nlp = nlp
        self.dataset = dataset
        self.split = split
        self.add_cols = [] if add_cols is None else add_cols
        self.input_col, self.target_col = summarization_name_mapping[self.args.dataset]
        self.ids2oracles = ids2oracles

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # beam = example['beam']
        sample = example['sample']

        source = sample['source']
        pred_abstracts = sample['abstract'].split('<cand>')
        # pred_abstract_rouges = list(map(float, example['abstract_rouges'].split(',')))
        # pred_extracts = example['extract'].split('<cand>')
        pred_extract_idxs = list(map(lambda x: x.replace(',', ''), sample['extract_idx'].split('<cand>')))
        pred_extract_rouges = list(map(float, sample['extract_rouges'].split(',')))
        pred_abstract_rouges = list(map(float, sample['extract_rouges'].split(',')))
        pred_implied_rouges = list(map(float, sample['implied_extract_rouges'].split(',')))
        num_cand = len(pred_abstracts)
        full_preds = [pred_extract_idxs[i] + '<sep>' + pred_abstracts[i] for i in range(num_cand)]

        avg_scores = [(a + b) / 2.0 for a, b in zip(pred_abstract_rouges, pred_extract_rouges)]

        oracle_order = np.argsort(-np.array(avg_scores))
        full_preds_ordered = [full_preds[rank] for rank in oracle_order]
        pred_extract_rouges_ordered = [pred_extract_rouges[rank] for rank in oracle_order]
        pred_abstract_rouges_ordered = [pred_abstract_rouges[rank] for rank in oracle_order]
        avg_scores_ordered = [avg_scores[rank] for rank in oracle_order]

        # Eliminate Candidates with Identical Scores
        uniq_idxs = [i for i in range(num_cand) if i == 0 or avg_scores_ordered[i] != avg_scores_ordered[i - 1]]
        full_preds_ordered = [full_preds_ordered[i] for i in uniq_idxs]
        pred_extract_rouges_ordered = [pred_extract_rouges_ordered[i] for i in uniq_idxs]
        pred_abstract_rouges_ordered = [pred_abstract_rouges_ordered[i] for i in uniq_idxs]
        avg_scores_ordered = [avg_scores_ordered[i] for i in uniq_idxs]

        return {
            'source': source,
            'targets': full_preds_ordered,
            'extract_scores': pred_extract_rouges_ordered,
            'abstract_scores': pred_abstract_rouges_ordered,
            'avg_scores': avg_scores_ordered,
        }
