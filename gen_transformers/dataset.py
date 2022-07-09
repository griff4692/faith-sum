import os
from string import punctuation
import ujson

import numpy as np
from nltk.corpus import stopwords
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
STOPWORDS = set(stopwords.words('english'))

from datasets import load_dataset, load_from_disk
from gen_transformers.data_utils import Seq2SeqCollate
from sum_constants import summarization_name_mapping


def remove_stopwords(tokens):
    return [w for w in tokens if not w.lower() in STOPWORDS and w not in punctuation and len(w.strip()) > 0]


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        if args.dataset == 'cnn_dailymail':
            data_dir = os.path.join(args.data_dir, args.dataset)
            self.dataset = load_from_disk(data_dir)
        else:
            self.dataset = load_dataset(args.dataset)
        self.tokenizer = tokenizer
        self.num_workers = 0  # 0 if args.debug else 4
        self.nlp = spacy.load('en_core_web_sm')

    def get_inverse_train_split(self, split, train_frac, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        max_examples = round(train_frac * len(split_dataset))
        n = len(split_dataset)
        trained_idxs = set(list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False))))
        idxs = list(sorted(set(range(n)) - trained_idxs))
        print(f'Using {len(idxs)} training examples set aside for re-ranking')
        print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
        split_dataset = split_dataset.select(idxs)

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, split)
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
            print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
            split_dataset = split_dataset.select(idxs)

        brio_candidates = None
        if self.args.add_sent_brio:
            oracle_brio = True  # Versus model predictions (doesn't mean we are training on BRIO)
            if oracle_brio:
                oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}_candidates_v2.json')
                with open(oracle_fn, 'r') as fd:
                    brio_candidates = ujson.load(fd)
            else:
                cand_fn = os.path.join(
                    self.args.data_dir, self.args.dataset, 'results', 'gen_extract_full', f'{split}_sample_outputs.csv'
                )
                cands = pd.read_csv(cand_fn)
                # TODO FORMAT THIS so that it looks like candidates
                brio_candidates = cands

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, split, brio_candidates=brio_candidates)
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
        return self.get_split('train', max_examples=max_examples)[0]

    def val_dataloader(self, max_examples=None):
        return self.get_split('validation', max_examples=max_examples or self.args.max_val_examples)[0]

    def test_dataloader(self, max_examples=None):
        return self.get_split('test', max_examples=max_examples)[0]


class SummarizationDataset(Dataset):
    def __init__(self, args, dataset, split, brio_candidates=None):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.split = split
        self.brio_candidates = brio_candidates
        self.input_col, self.target_col = summarization_name_mapping[self.args.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        dataset_id = example['id']
        target = example[self.target_col]

        # Let's get pre-computed oracle indices (locations of sentences included in oracle and oracle-abstract ROUGE)
        # Empirically, better performance from generating extracts in order in which they appear in source
        # Rather than by "relevance" as defined by ROUGE, for instance
        # Sort oracle order or not
        oracle_labels = np.sort(example['oracle_idxs'])
        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        source_annotated = example['source_annotated']
        input_ids = example['input_ids']
        if not self.args.add_sent_toks:
            # Use tokenizer min
            min_sent_id = input_ids[1]
            input_ids = [x for x in input_ids if x < min_sent_id]
        row = {
            'input_ids': input_ids,
            'labels': example['labels'],
            'source': source_annotated,
            'oracle_labels': oracle_labels,
            'reference': target,  # Use for evaluation
        }

        if self.args.add_sent_brio:
            candidates = [np.sort(x['extract_idx']) for x in self.brio_candidates[dataset_id]]
            if len(candidates) == 1:
                candidates.append(candidates[0])  # Revisit This (why do we have 1 candidate sometimes)
            row['oracle_cand_labels'] = candidates
        return row
