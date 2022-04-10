import numpy as np
np.random.seed(1992)
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import spacy

from datasets import load_dataset
from gen_transformers.data_utils import Seq2SeqCollate

from constants import summarization_name_mapping
from convert_abstractive_to_extractive import gain_selection


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, max_val_num=None):
        super().__init__()

        self.debug = args.debug
        if args.dataset == 'cnn_dailymail':
            self.dataset = load_dataset(args.dataset, '3.0.0')
        else:
            self.dataset = load_dataset(args.dataset)
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.tokenizer = tokenizer
        self.num_workers = 16
        self.max_val_num = max_val_num
        self.name = args.dataset
        self.nlp = spacy.load('en_core_web_sm')
        self.add_sent_toks = args.add_sent_toks
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.summary_style = args.summary_style

    def get_split(self, split, max_examples=None):
        split_dataset = self.dataset[split]
        if self.debug:
            max_examples = 128
        n = len(split_dataset)
        if max_examples is not None and max_examples < n:
            rand_idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            split_dataset = split_dataset.select(rand_idxs)
        split_dataset_pl = SummarizationDataset(
            split_dataset, split, self.nlp, self.max_input_length, dataset_name=self.name,
            add_sent_toks=self.add_sent_toks, summary_style=self.summary_style
        )
        add_cols = []
        if split != 'train' and self.summary_style == 'plan':
            add_cols.append('reference')
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=add_cols
        )
        kwargs = {
            'batch_size': self.per_device_train_batch_size if split == 'train' else self.per_device_eval_batch_size,
            'shuffle': split == 'train',
            'num_workers': 0 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(split_dataset_pl, **kwargs)

    def train_dataloader(self):
        return self.get_split('train')

    def val_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('validation', max_examples=max_examples)

    def test_dataloader(self, max_examples=None, add_cols=None):
        return self.get_split('test', max_examples=max_examples)


class SummarizationDataset(Dataset):
    def __init__(self, dataset, split, nlp, max_input_length, add_cols=None, dataset_name=None, add_sent_toks=False,
                 summary_style='abstract'):
        super(SummarizationDataset, self).__init__()
        self.nlp = nlp
        self.dataset = dataset
        self.split = split
        self.max_input_length = max_input_length
        self.add_cols = [] if add_cols is None else add_cols
        self.input_col, self.target_col = summarization_name_mapping[dataset_name]
        self.add_sent_toks = add_sent_toks
        self.summary_style = summary_style
        self.oracle_cutoff = 0.75  # TODO treat as hyper-parameter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        inputs = example[self.input_col]
        target = example[self.target_col]

        source_annotated = inputs
        if self.summary_style == 'abstract':
            target_annotated = target
            return {
                'source': source_annotated,
                'target': target_annotated
            }

        source_sents = list(self.nlp(inputs).sents)
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        if self.add_sent_toks:
            source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
        target_sents = list(self.nlp(target).sents)
        target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
        # Sort oracle order or not
        oracle = gain_selection(source_sents_tok, target_sents_tok, 5, lower=True, sort=True)
        oracle_idxs = oracle[0]
        target_prefix = ''.join([f'<s{i}>' for i in oracle_idxs]).strip()
        oracle_summary = ' '.join([str(source_sents[i]) for i in oracle_idxs])
        if self.summary_style == 'extract':
            if self.split == 'train':
                target_annotated = oracle_summary
            else:
                target_annotated = target  # We are evaluating on the abstractive summary
        elif self.summary_style == 'plan':
            target_annotated = target_prefix
        elif self.summary_style == 'plan_abstract':
            target_annotated = f'{target_prefix}<sep>{target}'
        elif self.summary_style == 'abstract_plan':
            target_annotated = f'{target}<sep>{target_prefix}'
        elif self.summary_style == 'hybrid_control':
            good_oracle = oracle[1]['rouge_1'] >= self.oracle_cutoff
            if self.split == 'train':
                prefix = '<extract>' if good_oracle else '<abstract>'
            else:
                # TODO We can do better than this ultimately for evaluation
                prefix = str(np.random.choice(['<abstract>', '<extract>'], size=(1, ))[0])
            target_annotated = oracle_summary if prefix == '<extract>' else target
            source_annotated = prefix + source_annotated
        output = {
            'source': source_annotated,
            'target': target_annotated,
        }
        if self.split != 'train' and self.summary_style == 'plan':
            output['reference'] = target
        return output
