import os
import itertools

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import spacy

from sum_constants import summarization_name_mapping


class Seq2SeqCollate:
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

    def get_split(self, split, max_examples=None):
        fn = os.path.join(self.args.data_dir, 'results', self.args.gen_experiment, f'{split}_outputs.csv')
        split_dataset = pd.read_csv(fn).to_dict('records')
        if self.args.debug and max_examples is None:
            max_examples = 128
        n = len(split_dataset)
        rand_idxs = None
        if max_examples is not None and max_examples < n:
            rand_idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = [split_dataset[i] for i in rand_idxs]
        add_cols = ['scores']
        # oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}.csv')
        # if not os.path.exists(oracle_fn):
        #     raise Exception(
        #         f'Please first run: python preprocess/extract_oracles.py '
        #         f'--dataset {self.args.dataset} --data_dir {self.args.data_dir}'
        #     )
        # print(f'Loading pre-computed oracle summaries from {oracle_fn}')
        # oracle_df = pd.read_csv(oracle_fn)
        # ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}

        split_dataset_pl = RankDataset(self.args, split_dataset, split, self.nlp, add_cols=add_cols)
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            add_cols=add_cols,
            split=split
        )
        batch_size = self.args.per_device_train_bs if split == 'train' else self.args.per_device_eval_bs
        kwargs = {
            'batch_size': batch_size,
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

        source = example['source']
        pred_abstracts = example['abstract'].split('<cand>')
        # pred_abstract_rouges = list(map(float, example['abstract_rouges'].split(',')))
        # pred_extracts = example['extract'].split('<cand>')
        pred_extract_idxs = list(map(lambda x: x.replace(',', ''), example['extract_idx'].split('<cand>')))
        pred_extract_rouges = list(map(float, example['extract_rouges'].split(',')))

        num_cand = len(pred_abstracts)
        full_preds = [pred_extract_idxs[i] + '<sep>' + pred_abstracts[i] for i in range(num_cand)]

        return {
            'source': source,
            'targets': full_preds,
            'scores': pred_extract_rouges,
        }
