import os

import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy

from datasets import load_dataset
from gen_transformers.data_utils import Seq2SeqCollate
from sum_constants import summarization_name_mapping
from preprocess.extract_oracles import convert_to_sents


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
        elif max_examples is None and split == 'train' and self.args.train_frac < 1:
            max_examples = round(self.args.train_frac * len(split_dataset))
        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = split_dataset.select(idxs)
        oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}_v2.csv')
        if not os.path.exists(oracle_fn):
            raise Exception(
                f'Please first run: python preprocess/extract_oracles.py '
                f'--dataset {self.args.dataset} --data_dir {self.args.data_dir}'
            )
        print(f'Loading pre-computed oracle summaries from {oracle_fn}')
        oracle_df = pd.read_csv(oracle_fn)
        ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, split, self.nlp, ids2oracles=ids2oracles)
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            split=split
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        inputs = example[self.input_col]
        target = example[self.target_col]
        source_annotated = inputs

        # Let's get pre-computed oracle indices (locations of sentences included in oracle and oracle-abstract ROUGE)
        oracle_obj = self.ids2oracles[example['id']]
        # Empirically, better performance from generating extracts in order in which they appear in source
        # Rather than by "relevance" as defined by ROUGE, for instance
        oracle_idxs = list(sorted(list(map(int, oracle_obj['sent_idxs'].split(',')))))
        # Sort oracle order or not
        oracle_labels = [i for i in oracle_idxs if i < self.args.max_num_sents]
        if 'extract' not in self.args.summary_style:
            oracle_labels = None

        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        if self.args.add_sent_toks:
            source_sents = convert_to_sents(inputs, self.nlp)
            source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        return {
            'source': source_annotated,
            'target': target,
            'oracle_labels': oracle_labels,
            'reference': target,  # Use for evaluation
        }
