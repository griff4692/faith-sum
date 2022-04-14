import os

import numpy as np
np.random.seed(1992)
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
import spacy
from string import punctuation

from datasets import load_dataset
from gen_transformers.data_utils import Seq2SeqCollate

from constants import summarization_name_mapping
from convert_abstractive_to_extractive import gain_selection

STOPWORDS = set(stopwords.words('english')).union(set(punctuation))


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

    def get_split(self, split, max_examples=None):
        split_dataset = self.dataset[split]
        if self.args.debug and max_examples is None:
            max_examples = 128
        n = len(split_dataset)
        if max_examples is not None and max_examples < n:
            rand_idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = split_dataset.select(rand_idxs)
        add_cols = []
        if split != 'train' and self.args.summary_style == 'plan':
            add_cols.append('reference')

        oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}.csv')
        print(f'Loading pre-computed oracle summaries from {oracle_fn}')
        oracle_df = pd.read_csv(oracle_fn)
        ids2oracles = {row['id']: row for row in oracle_df.to_dict('records')}

        split_dataset_pl = SummarizationDataset(
            self.args, split_dataset, split, self.nlp, add_cols=add_cols, ids2oracles=ids2oracles
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            add_cols=add_cols
        )
        batch_size = self.args.per_device_train_bs if split == 'train' else self.args.per_device_eval_bs
        kwargs = {
            'batch_size': batch_size,
            'shuffle': split == 'train',
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(split_dataset_pl, **kwargs)

    def train_dataloader(self):
        return self.get_split('train')

    def val_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('validation', max_examples=max_examples or self.args.max_val_examples)

    def test_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('test', max_examples=max_examples)


class SummarizationDataset(Dataset):
    def __init__(self, args, dataset, split, nlp, add_cols=None, ids2oracles=None):
        super(SummarizationDataset, self).__init__()
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
        inputs = example[self.input_col]
        target = example[self.target_col]

        source_annotated = inputs
        # For simple abstractive training, no oracle extracts / plans need to be included
        if self.args.summary_style == 'abstract':
            target_annotated = target
            return {
                'source': source_annotated,
                'target': target_annotated
            }

        # Let's get pre-computed oracle indices (locations of sentences included in oracle and oracle-abstract ROUGE)
        oracle_obj = self.ids2oracles[example['id']]
        oracle_idxs = list(map(int, oracle_obj['sent_idxs'].split(',')))

        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        source_sents = list(self.nlp(inputs).sents)
        if self.args.add_sent_toks:
            source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Sort oracle order or not
        target_prefix = ''.join([f'<s{i}>' for i in oracle_idxs]).strip()
        oracle_summary = ' '.join([str(source_sents[i]) for i in oracle_idxs])

        if self.args.summary_style == 'extract':
            if self.split == 'train':
                target_annotated = oracle_summary
            else:
                target_annotated = target  # We are evaluating on the abstractive summary
        elif self.args.summary_style == 'plan':
            target_annotated = target_prefix
        elif self.args.summary_style == 'plan_abstract':
            target_annotated = f'{target_prefix}<sep>{target}'
        elif self.args.summary_style == 'abstract_plan':
            target_annotated = f'{target}<sep>{target_prefix}'
        elif self.args.summary_style == 'hybrid_control':
            if self.split == 'train':
                avg_oracle_rouge = (oracle_obj['rouge_1'] + oracle_obj['rouge_2']) / 2.0
                good_oracle = avg_oracle_rouge >= self.args.oracle_cutoff
                prefix = '<extract>' if good_oracle else '<abstract>'
            else:
                # TODO We can do better than this ultimately for evaluation
                prefix = '<extract>'  # <abstract>
                # prefix = str(np.random.choice(['<abstract>', '<extract>'], size=(1,))[0])

            target_annotated = oracle_summary if prefix == '<extract>' and self.split == 'train' else target
            source_annotated = prefix + source_annotated
        output = {
            'source': source_annotated,
            'target': target_annotated,
        }
        if self.split != 'train' and self.args.summary_style == 'plan':
            output['reference'] = target
        return output
