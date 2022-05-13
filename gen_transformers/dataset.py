import os

import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
from scipy.special import softmax
from tqdm import tqdm

from datasets import load_dataset
from gen_transformers.data_utils import Seq2SeqCollate
from sum_constants import summarization_name_mapping
from preprocess.extract_oracles import convert_to_sents


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
        add_cols = ['reference']
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
        self.temperature = 5.0

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

    def get_plan_q(self, oracle_obj):
        r1s = [float(x) for x in oracle_obj['rouge1_history'].split('|')[0].split(',')]
        r2s = [float(x) for x in oracle_obj['rouge2_history'].split('|')[0].split(',')]
        avg_rs = np.array([(a + b) / 2.0 for a, b in zip(r1s, r2s)])
        min_rs = min(avg_rs)
        max_rs = max(avg_rs)
        if max_rs == min_rs:
            scaled_rs = avg_rs
        else:
            scaled_rs = (avg_rs - min_rs) / (max_rs - min_rs)
        plan_q = softmax(self.temperature * scaled_rs)
        return plan_q

    def __getitem__(self, idx):
        example = self.dataset[idx]
        inputs = example[self.input_col]
        target = example[self.target_col]

        untouched_target = target  # Original untouched reference

        source_annotated = inputs
        # For simple abstractive training, no oracle extracts / plans need to be included
        if self.args.summary_style == 'abstract':
            target_annotated = target
            return {
                'source': source_annotated,
                'target': target_annotated,
                'reference': untouched_target  # Same as target here but not always true
            }

        # Let's get pre-computed oracle indices (locations of sentences included in oracle and oracle-abstract ROUGE)
        oracle_obj = self.ids2oracles[example['id']]
        oracle_idxs = list(map(int, oracle_obj['sent_idxs'].split(',')))

        r1s = [float(x) for x in oracle_obj['rouge1_history'].split('|')[0].split(',')]
        r2s = [float(x) for x in oracle_obj['rouge2_history'].split('|')[0].split(',')]
        avg_rs = np.array([(a + b) / 2.0 for a, b in zip(r1s, r2s)])
        sent_priority = avg_rs

        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        # source_sents = list(self.nlp(inputs).sents)
        source_sents = convert_to_sents(inputs, self.nlp)
        if self.args.add_sent_toks:
            source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        # Sort oracle order or not
        target_prefix = ''.join([f'<s{i}>' for i in oracle_idxs]).strip()
        oracle_summary = ' '.join([str(source_sents[i]) for i in oracle_idxs])
        plan_labels = None
        plan_q = None

        if self.args.summary_style == 'extract':
            if self.split == 'train':
                target_annotated = oracle_summary
            else:
                target_annotated = target  # We are evaluating on the abstractive summary
        elif self.args.summary_style == 'plan':
            target_annotated = target_prefix
        elif self.args.summary_style == 'plan_abstract':
            target_annotated = f'{target_prefix}<sep>{target}'
        elif self.args.summary_style == 'score_abstract':
            target_annotated = target
            plan_labels = [i for i in oracle_idxs if i < self.args.max_num_sents]
            assert len(plan_labels) >= 1
            plan_q = self.get_plan_q(oracle_obj)
        elif self.args.summary_style == 'score':
            target_annotated = None  # No generation, just sentence scoring and selection for extractive summarization
            plan_labels = [i for i in oracle_idxs if i < self.args.max_num_sents]
            assert len(plan_labels) >= 1
            plan_q = self.get_plan_q(oracle_obj)
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
        return {
            'source': source_annotated,
            'target': target_annotated,
            'plan_labels': plan_labels,
            'plan_q': plan_q,
            'sent_priority': sent_priority,
            'reference': untouched_target,  # Use for evaluation
        }
