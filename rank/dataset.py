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
from eval.rouge_metric import RougeMetric


def compute_rouge(rouge_metric, generated, gold):
    outputs = rouge_metric.evaluate_batch([generated], [gold], aggregate=True)['rouge']
    f1s = []
    for rouge_type in ['1', '2', 'L']:
        fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
        f1s.append(fscore)
    return f1s


class RankCollate:
    def __init__(self, tokenizer, max_input_length=512, add_cols=None, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.pad_id = tokenizer.pad_token_id
        self.add_cols = [] if add_cols is None else add_cols
        self.split = split

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        batch_size = len(batch_list)
        abstracts_flat = list(itertools.chain(*[x['abstracts'] for x in batch_list]))
        num_cands = len(batch_list[0]['abstracts'])
        sources_flat = list(itertools.chain(*[[x['source']] * num_cands for x in batch_list]))
        batch_inputs = self.tokenizer(
            abstracts_flat,
            sources_flat,
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        scores = [x['scores'] for x in batch_list]

        norm_scores = np.zeros([batch_size, num_cands])
        for batch_idx, score_arr in enumerate(scores):
            score_arr = np.array(score_arr)
            normed = (score_arr - min(score_arr)) / (max(score_arr) - min(score_arr))
            norm_scores[batch_idx] = normed

        norm_scores = torch.from_numpy(norm_scores)
        return {'inputs': batch_inputs, 'scores': scores, 'score_dist': norm_scores}


class RankDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 16
        self.nlp = spacy.load('en_core_web_sm')

        beam_fn = os.path.join(self.args.data_dir, 'results', 'gen_abstract_full', 'validation_beam_outputs.csv')
        sample_fn = os.path.join(
            self.args.data_dir, 'results', self.args.gen_experiment, 'from_sample_extract.csv'
        )

        beam_df = pd.read_csv(beam_fn)
        sample_df = pd.read_csv(sample_fn)

        avail_idxs = sample_df['dataset_idx'].unique().tolist()
        beam_df = beam_df[beam_df['dataset_idx'].isin(set(avail_idxs))]

        sample_records = {record['dataset_idx']: record for record in sample_df.to_dict('records')}
        beam_records = {record['dataset_idx']: record for record in beam_df.to_dict('records')}
        np.random.seed(1992)
        train_frac = 0.8
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
        split_dataset_pl = RankDataset(self.args, split_dataset, split, self.nlp)
        collate_fn = RankCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            split=split
        )
        kwargs = {
            'batch_size': self.args.batch_size,
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
        self.rouge_metric = RougeMetric()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        beam = example['beam']
        sample = example['sample']

        source = sample['source']
        reference = sample['reference']
        pred_abstracts = sample['from_extract_abstract'].split('<cand>')
        pred_abstracts.insert(0, beam['abstract'])

        rouges = []
        for abstract in pred_abstracts:
            rouges.append(compute_rouge(self.rouge_metric, abstract, reference)[0])

        oracle_order = np.argsort(-np.array(rouges))
        pred_abstracts = [pred_abstracts[i] for i in oracle_order]
        # pred_abstracts[0] = 'WINNER: '
        rouges = [rouges[i] for i in oracle_order]
        if len(pred_abstracts) != 17:
            print(len(pred_abstracts))
            additional_copies = [pred_abstracts[-1]] * (17 - len(pred_abstracts))
            additional_scores = [rouges[-1]] * (17 - len(pred_abstracts))
            pred_abstracts = pred_abstracts + additional_copies
            rouges = rouges + additional_scores
        return {
            'source': source,
            'abstracts': pred_abstracts,
            'scores': rouges,
        }
