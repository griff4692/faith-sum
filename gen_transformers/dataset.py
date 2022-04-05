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
            add_sent_toks=self.add_sent_toks
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
        )
        kwargs = {
            'batch_size': self.per_device_train_batch_size if split == 'train' else self.per_device_eval_batch_size,
            'shuffle': split == 'train',
            'num_workers': 1 if self.debug else self.num_workers,
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
    def __init__(self, dataset, split, nlp, max_input_length, add_cols=None, dataset_name=None, add_sent_toks=False):
        super(SummarizationDataset, self).__init__()
        self.nlp = nlp
        self.dataset = dataset
        self.split = split
        self.max_input_length = max_input_length
        self.add_cols = [] if add_cols is None else add_cols
        self.input_col, self.target_col = summarization_name_mapping[dataset_name]
        self.add_sent_toks = add_sent_toks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        inputs = example[self.input_col]
        target = example[self.target_col]

        source_annotated, target_annotated = inputs, target
        if self.add_sent_toks:
            source_sents = list(self.nlp(inputs).sents)
            source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
            target_sents = list(self.nlp(target).sents)
            target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
            source_annotated = ''.join([f'<s{i}> {s}' for i, s in enumerate(source_sents)])
            # Sort oracle order or not
            oracle = gain_selection(source_sents_tok, target_sents_tok, 5, lower=True, sort=True)
            target_prefix = ''.join([f'<s{i}>' for i in oracle[0]]).strip()
            target_annotated = f'{target_prefix}<sep>{target}'  # <sep>
        return {
            'source': source_annotated,
            'target': target_annotated
        }
