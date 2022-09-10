import os
import itertools

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from datasets import load_from_disk

from sum_constants import summarization_name_mapping
from eval.rouge_metric import RougeMetric


def compute_rouge(rouge_metric, generated, gold):
    outputs = rouge_metric.evaluate_batch([generated], [gold], aggregate=True)['rouge']
    f1s = []
    for rouge_type in ['1', '2', 'L']:
        fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
        f1s.append(fscore)
    return f1s


def extract_indicators(cls_mask, sent_idx_to_mask):
    """
    :param cls_mask: indications of where sentences tokens are
    :param sent_idx_to_mask: which sentences to mask (the sentence order, not location in cls_mask)
    :return:
    """
    extract_indicators = torch.zeros_like(cls_mask, device=cls_mask.device).long()
    sent_locs = cls_mask.nonzero()[:, 0]
    max_seq_len = len(cls_mask)
    num_sents = len(sent_locs)
    for sent_idx, sent_loc in enumerate(sent_locs):
        sent_loc = sent_loc.item()
        end_loc = sent_locs[sent_idx + 1].item() if sent_idx + 1 < num_sents else max_seq_len
        if sent_idx in sent_idx_to_mask:
            extract_indicators[sent_loc:end_loc] = 1
    return extract_indicators


class RankCollate:
    def __init__(self, tokenizer, max_input_length=512, add_cols=None, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.pad_id = tokenizer.pad_token_id
        self.add_cols = [] if add_cols is None else add_cols
        self.split = split
        additional_ids = self.tokenizer.additional_special_tokens_ids
        self.special_id_min = 999999 if len(additional_ids) == 0 else min(self.tokenizer.additional_special_tokens_ids)

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        batch_size = len(batch_list)
        # abstracts_flat = list(itertools.chain(*[x['abstracts'] for x in batch_list]))
        extract_idxs_flat = list(itertools.chain(*[x['extract_idxs'] for x in batch_list]))
        num_cands = len(batch_list[0]['extract_idxs'])
        sources_flat = list(itertools.chain(*[[x['source']] * num_cands for x in batch_list]))
        batch_inputs = self.tokenizer(
            # abstracts_flat,
            sources_flat,
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        cls_mask = batch_inputs['input_ids'] >= self.special_id_min
        extract_indicator_ids = []
        n = len(cls_mask)
        for flat_idx in range(n):
            extract_indicator_ids.append(
                extract_indicators(cls_mask[flat_idx], extract_idxs_flat[flat_idx])
            )
        extract_indicator_ids = torch.stack(extract_indicator_ids)
        extract_indicator_ids.masked_fill_(batch_inputs['attention_mask'] == 0, 0)
        batch_inputs['extract_indicators'] = extract_indicator_ids
        scores = [x['scores'] for x in batch_list]
        features_flat = torch.from_numpy(np.array(list(itertools.chain(*[x['features'] for x in batch_list])))).float()
        batch_inputs['classifier_features'] = features_flat
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
        self.dataset = load_from_disk(os.path.join(args.data_dir, args.dataset))

    def get_split(self, split, max_examples=None):
        split_fn = os.path.join(
            self.args.data_dir, 'results', self.args.gen_experiment, f'{split}_sample_outputs.csv'
        )
        split_df = pd.read_csv(split_fn)
        split_data = self.dataset[split]

        annotated_source = split_data['source_annotated']
        oracle_idxs = split_data['oracle_idxs']
        split_dataset = []
        for record in split_df.to_dict('records'):
            try:
                cand_extract_idx = [[int(x) for x in idx.split(',')] for idx in record['extract_idx'].split('<cand>')]
            except Exception as e:
                print(f'Skipping: {e}')
                continue

            extract_rouges = [float(x) for x in record['extract_rouges'].split(',')]
            if min(extract_rouges) == max(extract_rouges):
                print(f'Min and max ROUGE is the same: {min(extract_rouges)}. Skipping')
                continue

            split_dataset.append({
                'dataset_idx': record['dataset_idx'],
                'annotated_source': annotated_source[record['dataset_idx']],
                'reference': record['reference'],
                'oracle_extract_idx': oracle_idxs[record['dataset_idx']],
                'cand_extract_idx': cand_extract_idx,
                'num_candidates': len(cand_extract_idx),
                'cand_extract_rouges': extract_rouges,
            })

        # Removing incomplete extract candidates
        max_candidates = max([x['num_candidates'] for x in split_dataset])
        split_dataset_filt = [
            x for x in split_dataset if x['num_candidates'] == max_candidates
        ]
        print(f'{len(split_dataset)}/{len(split_dataset_filt)} have the max number of candidates={max_candidates}.')

        if self.args.debug and max_examples is None:
            max_examples = 128
        n = len(split_dataset_filt)
        rand_idxs = None
        if max_examples is not None and max_examples < n:
            rand_idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset_filt = [split_dataset_filt[i] for i in rand_idxs]
        split_dataset_pl = RankDataset(self.args, split_dataset_filt, split, self.nlp)
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
        self.num_candidates = 17

    def __len__(self):
        return len(self.dataset)

    def build_features(self, extract_idxs, beam_frac, source_sents):
        num_sents = len(extract_idxs)
        num_source_sents = len(source_sents)
        sent_frac = num_sents / num_source_sents
        extract = [source_sents[i] for i in extract_idxs]
        source_toks = ' '.join(source_sents).split(' ')
        num_source_toks = len(source_toks)
        extract_toks = ' '.join(extract).split(' ')
        num_extract_toks = len(extract_toks)
        token_frac = num_extract_toks  / num_source_toks
        average_pos = np.mean(extract_idxs) / len(source_sents)
        return [
            beam_frac,
            num_sents,
            sent_frac,
            token_frac,
            average_pos
        ]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        source = example['annotated_source']
        oracle_extract_idx = example['oracle_extract_idx']
        cand_extract_idx = example['cand_extract_idx']
        n = len(cand_extract_idx)
        cand_extract_rouges = example['cand_extract_rouges']

        # Sort the extract indices
        beam_priority = np.argsort(-np.array(cand_extract_rouges))
        cand_extract_idx_ordered = [cand_extract_idx[i] for i in beam_priority]
        cand_extract_rouges_ordered = [cand_extract_rouges[i] for i in beam_priority]

        features = []
        import regex as re
        source_sents = re.split(r'(<s\d+>)', source)
        source_sents = [
            source_sents[i + 1].strip() for i in range(len(source_sents))
            if re.match(r'<s\d+>', source_sents[i]) is not None
        ]
        for cand_idx in range(n):
            extract_idxs = cand_extract_idx_ordered[cand_idx]
            beam_idx = beam_priority[cand_idx]
            beam_frac = (beam_idx + 1) / n
            features.append(self.build_features(extract_idxs, beam_frac, source_sents))

        return {
            'source': source,
            'features': features,
            'extract_idxs': cand_extract_idx_ordered,
            'scores': cand_extract_rouges_ordered,
        }
