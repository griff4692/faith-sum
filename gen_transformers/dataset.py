import ujson
import os
from string import punctuation
import regex as re

import numpy as np
from nltk.corpus import stopwords
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import spacy
STOPWORDS = set(stopwords.words('english'))

from datasets import load_from_disk
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
        if 'pegasus' in args.hf_model:
            data_dir = os.path.join(args.data_dir, args.dataset + '_pegasus')
        else:
            data_dir = os.path.join(args.data_dir, args.dataset)
        print(f'Loading data from {data_dir}')
        self.dataset = load_from_disk(data_dir)
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 8
        self.nlp = spacy.load('en_core_web_sm')

        # if self.args.is_word_brio and self.args.use_regular_candidates:
        #     candidates_fn = '/nlp/projects/faithsum/cnn_brio_outputs.json'
        #     print(f'Loading {candidates_fn}')
        #     with open(candidates_fn, 'r') as fd:
        #         self.brio_candidates = ujson.load(fd)

    def get_train_chunk(self, chunk, num_chunks, **dataloader_kwargs):
        split_dataset = self.dataset['train']
        n = len(split_dataset)

        all_idxs = list(range(n))
        chunk_idxs = np.array_split(all_idxs, num_chunks)[chunk]
        print(f'Using {len(chunk_idxs)} training examples set for chunk {chunk}/{num_chunks}')
        print(f'First {min(3, len(chunk_idxs))} idxs: {chunk_idxs[:min(3, len(chunk_idxs))]}')
        split_dataset = split_dataset.select(chunk_idxs)

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, self.tokenizer, 'train')
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            split='train',
        )
        kwargs = {
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        kwargs.update(**dataloader_kwargs)
        return DataLoader(split_dataset_pl, **kwargs), chunk_idxs

    def get_split(self, split, max_examples=None, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        if self.args.debug and max_examples is None:
            max_examples = 128
        elif max_examples is None and split == 'train' and self.args.train_frac < 1:
            max_examples = round(self.args.train_frac * len(split_dataset))

        brio_candidates = None
        if self.args.add_brio_loss:
            out_dir = os.path.join(self.args.data_dir, self.args.dataset, 'oracle')
            bert_suffix = '_bert' if 'bert' in self.args.oracle_col else ''
            out_fn = os.path.join(out_dir, f'{split}_candidates{bert_suffix}.json')
            with open(out_fn, 'r') as fd:
                candidates = ujson.load(fd)

            brio_candidates = {}
            for dataset_id, cands in candidates.items():
                if 'bert' in self.args.oracle_col:
                    scores_ordered = [x['mean_f1'] for x in cands]
                else:
                    scores_ordered = [x['bertscore_f1'] for x in cands]

                extract_idxs_ordered = [x['extract_idx'] for x in cands]
                for i in range(1, len(scores_ordered)):  # Assert it's pre-sorted by ROUGE
                    assert scores_ordered[i - 1] >= scores_ordered[i]
                if len(cands) < 2:
                    continue
                brio_candidates[dataset_id] = [extract_idxs_ordered, scores_ordered]

            # Filter dataset to only include ones with BRIO candidates generated or 'oracled'
            available_keys = set(list(brio_candidates.keys()))
            valid_hf_ids = [i for i, dataset_idx in enumerate(split_dataset['id']) if dataset_idx in available_keys]
            print(f'Filtering out for {len(valid_hf_ids)}/{len(split_dataset)} contrastive BRIO examples')
            split_dataset = split_dataset.select(valid_hf_ids)

        # if self.args.add_brio_loss:
        #     if self.args.is_word_brio:
        #         if self.args.use_regular_candidates:
        #             max_brio_candidates = 4
        #             brio_candidates_raw = self.brio_candidates[split]
        #             brio_candidates = {}
        #             for dataset_id, arr in brio_candidates_raw.items():
        #                 candidates = arr[0][:max_brio_candidates]
        #                 rouges = np.array(arr[1][:max_brio_candidates])
        #                 priority = np.argsort(-rouges)
        #                 cands_ordered = [candidates[i] for i in priority]
        #                 rouges_ordered = [rouges[i] for i in priority]
        #                 brio_candidates[dataset_id] = [cands_ordered, rouges_ordered]
        #         else:
        #             extract_fn = '/nlp/projects/faithsum/results/add_doc/'
        #             cand_fn = extract_fn + f'{split}_from_sample_w_diverse_extract_4.csv'
        #             print(f'Loading in over-generated candidates from {cand_fn}')
        #             cands = pd.read_csv(cand_fn)
        #             brio_candidates = {}
        #             dataset_ids = list(split_dataset['id'])
        #             for record in tqdm(cands.to_dict('records'), total=len(cands), desc='Sorting them:'):
        #                 rouges = np.array(list(map(float, record['from_extract_rouges'].split('<cand>'))))
        #                 priority = np.argsort(-rouges)
        #                 from_extract_abstracts = record['from_extract_abstract'].split('<cand>')
        #                 cands_ordered = [from_extract_abstracts[i] for i in priority]
        #                 rouges_ordered = [rouges[i] for i in priority]
        #                 brio_candidates[dataset_ids[record['dataset_idx']]] = [cands_ordered, rouges_ordered]
        #     else:
        #         cand_fn = os.path.join(
        #             self.args.data_dir, 'results',
        #             self.args.brio_experiment, f'{split}_sample_w_diverse_outputs.csv'
        #         )
        #         print(f'Loading in over-generated candidates from {cand_fn}')
        #         cands = pd.read_csv(cand_fn)
        #         brio_candidates = defaultdict(list)
        #         dataset_ids = list(split_dataset['id'])
        #         for record in tqdm(cands.to_dict('records'), total=len(cands), desc='Sorting them:'):
        #             extract_rouges = np.array(list(map(float, record['extract_rouges'].split(','))))
        #             try:
        #                 extract_idxs = [
        #                     [int(x) for x in idx_str.split(',')] for idx_str in record['extract_idx'].split('<cand>')
        #                 ]
        #             except:
        #                 print('Cannot parse empty extract')
        #                 continue
        #
        #             assert len(extract_rouges) == len(extract_idxs)
        #             priority = np.argsort(-extract_rouges)
        #             extract_idxs_ordered = [extract_idxs[i] for i in priority]
        #             extract_rouges_ordered = [extract_rouges[i] for i in priority]
        #
        #             # De-Duplicate
        #             extract_idxs_uniq = [
        #                 extract_idx for i, extract_idx in enumerate(
        #                     extract_idxs_ordered
        #                 ) if i == 0 or extract_idxs_ordered[i - 1] != extract_idx
        #             ]
        #
        #             extract_rouges_uniq = [
        #                 extract_rouges_ordered[i] for i, extract_idx in enumerate(
        #                     extract_idxs_ordered
        #                 ) if i == 0 or extract_idxs_ordered[i - 1] != extract_idx
        #             ]
        #
        #             if len(extract_rouges_uniq) < 2:
        #                 print(f'Only {len(extract_rouges_uniq)} unique candidate provided for this example.')
        #                 continue
        #
        #             brio_candidates[dataset_ids[record['dataset_idx']]] = [extract_idxs_uniq, extract_rouges_uniq]

        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
            split_dataset = split_dataset.select(idxs)

        if split == 'train' and self.args.oracle_drop_p > 0:
            oracle_rouge_1 = split_dataset['oracle_rouge1']
            oracle_rouge_2 = split_dataset['oracle_rouge2']
            avg_rouge = [(a + b) / 2.0 for (a, b) in zip(oracle_rouge_1, oracle_rouge_2)]
            priority = np.argsort(avg_rouge)
            drop_n = round(len(split_dataset) * self.args.oracle_drop_p)
            print(f'Filtering out {drop_n} training examples with lowest oracle score.')
            keep_idxs = list(sorted(priority[drop_n:]))
            split_dataset = split_dataset.select(keep_idxs)

        split_dataset_pl = SummarizationDataset(
            self.args, split_dataset, self.tokenizer, split, brio_candidates=brio_candidates
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            split=split,
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
    def __init__(self, args, dataset, tokenizer, split, brio_candidates=None):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
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
        oracle_labels = np.sort(example[self.args.oracle_column])
        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        source_annotated = example['source_annotated']
        input_ids = example['input_ids']
        if not self.args.add_sent_toks:
            # Use tokenizer min
            first_sent_idx = 0 if 'pegasus' in self.args.hf_model else 1
            if 'pegasus' in self.args.hf_model:  # Can remove once this clears.
                assert example['input_ids'][0] == 96103
            min_sent_id = input_ids[first_sent_idx]
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
            candidates, scores = self.brio_candidates[dataset_id].copy()  # We modify it in place so let's insert

            if len(candidates) > self.args.max_brio_candidates:
                candidates = candidates[:self.args.max_brio_candidates]
                scores = scores[:self.args.max_brio_candidates]
            # for i in range(len(candidates)):
            #     if type(candidates[i]) == dict:
            #         if i < len(candidates) - 1:
            #             assert candidates[i]['mean_f1'] >= candidates[i + 1]['mean_f1']
            #         candidates[i] = list(sorted(candidates[i]['extract_idx']))

            scores = np.array(scores)
            norm_scores = (scores - min(scores)) / (max(scores) - min(scores))

            oracle_in_list = any([
                list(oracle_labels) == cand for cand in candidates
            ])
            # If we want to include the oracle in the Gold
            if not oracle_in_list and self.args.include_gold:  # If the model didn't already generate the oracle
                # Add Gold Label as the 'most positive'
                candidates.insert(0, list(oracle_labels))
                norm_scores.insert(0, 1)

            # if self.args.is_word_brio:
            #     with self.tokenizer.as_target_tokenizer():
            #         brio_word_labels = np.array(self.tokenizer(
            #             candidates, max_length=1024, truncation=True, padding='longest'
            #         )['input_ids'], dtype=np.int64)
            #         brio_word_labels[np.where(brio_word_labels == self.tokenizer.pad_token_id)] = -100
            #     row['brio_word_labels'] = brio_word_labels   # brio_word_labels (change back if you need this)
            # else:
            row['brio_sent_labels'] = candidates
            row['brio_norm_scores'] = norm_scores
        return row
