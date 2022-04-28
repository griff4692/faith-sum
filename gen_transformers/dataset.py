import os

import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
from tqdm import tqdm

from datasets import load_dataset
from gen_transformers.data_utils import Seq2SeqCollate

from sum_constants import summarization_name_mapping


def remove_sent_from_plan(oracle_idxs):
    list_idx_to_remove = int(np.random.choice(np.arange(len(oracle_idxs)), size=(1,))[0])
    remove_idxs = oracle_idxs[:list_idx_to_remove] + oracle_idxs[list_idx_to_remove + 1:]
    return ''.join([f'<s{i}>' for i in remove_idxs]).strip()


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        if args.dataset == 'cnn_dailymail':
            self.dataset = load_dataset(args.dataset, '3.0.0')
        else:
            self.dataset = load_dataset(args.dataset)
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 16
        self.nlp = spacy.load('en_core_web_sm')

    def get_split(self, split, max_examples=None, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        if self.args.debug and max_examples is None:
            max_examples = 128
        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = split_dataset.select(idxs)
        oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}.csv')
        if not os.path.exists(oracle_fn):
            raise Exception(
                f'Please first run: python preprocess/extract_oracles.py '
                f'--dataset {self.args.dataset} --data_dir {self.args.data_dir}'
            )
        print(f'Loading pre-computed oracle summaries from {oracle_fn}')
        oracle_df = pd.read_csv(oracle_fn)
        ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, split, self.nlp, ids2oracles=ids2oracles)
        add_cols = ['reference']
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            add_cols=add_cols,
            split=split,
            verbose=self.args.debug
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
    def __init__(self, args, dataset, split, nlp, ids2oracles=None):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.nlp = nlp
        self.dataset = dataset
        self.split = split
        self.input_col, self.target_col = summarization_name_mapping[self.args.dataset]
        self.ids2oracles = ids2oracles

        if self.args.oracle_filter and split == 'train':  # Only filter for training data
            keep_idxs = []
            n = len(self.dataset)
            print(f'Filtering training dataset for high-quality oracles only: avg R1/R2 >= {self.args.oracle_cutoff}')
            for idx, example in tqdm(enumerate(self.dataset), total=n):
                oracle_obj = self.ids2oracles[example['id']]
                avg_oracle_rouge = (oracle_obj['rouge_1'] + oracle_obj['rouge_2']) / 2.0
                good_oracle = avg_oracle_rouge >= self.args.oracle_cutoff
                if good_oracle:
                    keep_idxs.append(idx)
            self.dataset = self.dataset.select(keep_idxs)
            print(f'Truncated training data from {n} to {len(self.dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        inputs = example[self.input_col]
        target = example[self.target_col]

        untouched_target = target  # Original untouched reference

        # Let's get pre-computed oracle indices (locations of sentences included in oracle and oracle-abstract ROUGE)
        oracle_obj = self.ids2oracles[example['id']]
        oracle_idxs = list(map(int, oracle_obj['sent_idxs'].split(',')))

        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        # source_sents = list(self.nlp(inputs).sents)
        # if len(source_sents) > self.args.max_num_sents:
        #     print(f'Taking the first {self.args.max_num_sents} sentences.')
        #     source_sents = source_sents[:self.args.max_num_sents]

        # Insert [CLS] tokens before each sentence
        # source_annotated = ''.join([f'<s> {s}' for i, s in enumerate(source_sents)])
        source_annotated = inputs  # ' '.join(source_sents)

        # Sort oracle order or not
        plan_labels = [i for i in oracle_idxs if i < self.args.max_num_sents]
        target_annotated = target
        # oracle_summary = ' '.join([str(source_sents[i]) for i in oracle_idxs if i < max_num_sents])
        output = {
            'source': source_annotated,
            'target': target_annotated,
            'reference': untouched_target,  # Use for evaluation
            'plan_labels': plan_labels,
        }
        return output
