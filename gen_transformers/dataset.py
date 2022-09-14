import ujson
import os
from string import punctuation
from collections import defaultdict
import regex as re

import numpy as np
from nltk.corpus import stopwords
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
STOPWORDS = set(stopwords.words('english'))

from datasets import load_dataset, load_from_disk
from gen_transformers.data_utils import Seq2SeqCollate
from preprocess.helpers import _get_ngrams
from sum_constants import summarization_name_mapping


def remove_stopwords(tokens):
    return [w for w in tokens if not w.lower() in STOPWORDS and w not in punctuation and len(w.strip()) > 0]


def get_sent_ngrams(source_annotated):
    tps = re.split(r'(<s\d+>)', source_annotated)
    source_sents = []
    for tp_idx, tp in enumerate(tps):
        if re.match(r'(<s\d+>)', tp) is not None:
            source_sents.append(tps[tp_idx + 1].strip())

    num_sents = len(re.findall(r'(<s\d+>)', source_annotated))
    assert len(source_sents) == num_sents
    def get_ngrams(sent):
        toks = list(map(lambda x: x.lower(), sent.split(' ')))
        return [_get_ngrams(1, toks), _get_ngrams(2, toks), _get_ngrams(3, toks)]
    source_ngrams = list(map(get_ngrams, source_sents))
    return source_ngrams


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

    def get_inverse_train_split(self, split, train_frac, max_examples=None, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        n = len(split_dataset)
        if train_frac > 0:
            train_num = round(train_frac * len(split_dataset))
            trained_idxs = set(list(np.sort(np.random.choice(np.arange(n), size=(train_num, ), replace=False))))
            idxs = list(sorted(set(range(n)) - trained_idxs))
        else:
            idxs = list(range(n))

        if max_examples is not None and max_examples < len(idxs):
            print(f'Subsampling for maximum of {max_examples}')
            idxs = list(np.sort((np.random.choice(idxs, size=(max_examples,), replace=False))))

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

        brio_candidates = None
        if self.args.add_brio_loss:
            if self.args.brio_experiment is None:
                oracle_fn = os.path.join(self.args.data_dir, self.args.dataset, 'oracle', f'{split}_candidates_v2.json')
                with open(oracle_fn, 'r') as fd:
                    brio_candidates = ujson.load(fd)
                    n = len(brio_candidates)
                    brio_candidates = {k: v for k, v in brio_candidates.items() if len(v) >= 2}
                    filt_n = len(brio_candidates)
                    print(f'{filt_n}/{n} have more than 1 candidate provided for {split} set')
            else:
                cand_fn = os.path.join(
                    self.args.data_dir, 'results',
                    self.args.brio_experiment, f'{split}_sample_outputs.csv'
                )
                cands = pd.read_csv(cand_fn)
                brio_candidates = defaultdict(list)
                dataset_ids = list(split_dataset['id'])
                for record in cands.to_dict('records'):
                    extract_rouges = np.array(list(map(float, record['extract_rouges'].split(','))))
                    extract_idxs = [[int(x) for x in idx_str.split(',')] for idx_str in record['extract_idx'].split('<cand>')]

                    assert len(extract_rouges) == len(extract_idxs)
                    priority = np.argsort(-extract_rouges)
                    extract_idxs_ordered = [extract_idxs[i] for i in priority]

                    # De-Duplicate
                    extract_idxs_uniq = [
                        extract_idx for i, extract_idx in enumerate(
                            extract_idxs_ordered
                        ) if i == 0 or extract_idxs_ordered[i - 1] != extract_idx
                    ]

                    brio_candidates[dataset_ids[record['dataset_idx']]] = extract_idxs_uniq
            # Filter dataset to only include ones with BRIO candidates generated or "oracled"
            available_keys = set(list(brio_candidates.keys()))
            valid_hf_ids = [i for i, dataset_idx in enumerate(split_dataset['id']) if dataset_idx in available_keys]
            print(f'Filtering out for {len(valid_hf_ids)} contrastive BRIO examples')
            split_dataset = split_dataset.select(valid_hf_ids)

        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
            split_dataset = split_dataset.select(idxs)

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
        return self.get_split('train', max_examples=None)[0]

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
            'source_ngrams': get_sent_ngrams(source_annotated)
        }

        if self.args.add_brio_loss:
            candidates = self.brio_candidates[dataset_id]
            for i in range(len(candidates)):
                if type(candidates[i]) == dict:
                    if i < len(candidates) - 1:
                        assert candidates[i]['mean_f1'] >= candidates[i + 1]['mean_f1']
                    candidates[i] = list(sorted(candidates[i]['extract_idx']))

            # Add Gold Label as the 'most positive'
            candidates.insert(0, oracle_labels)

            # num_cand = len(candidates)
            # max_len = max([len(x) for x in candidates])
            # brio_labels = np.zeros([num_cand, max_len], dtype=np.int64)
            # brio_labels.fill(-100)
            # for cand_idx in range(num_cand):
            #     brio_labels[cand_idx, :len(candidates[cand_idx])] = candidates[cand_idx]
            brio_labels = candidates
            row['brio_labels'] = brio_labels
        return row
