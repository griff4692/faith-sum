import os
from pathlib import Path

import itertools
import nltk
import numpy as np
import torch


def postprocess_text(texts):
    return ['\n'.join(nltk.sent_tokenize(text.strip())) for text in texts]


def source_from_ids(input_ids, nlp, tokenizer):
    source = tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
    source_docs = [list(nlp(x).sents) for x in source]
    source_doc_sents_tok = [
        [[str(token.text) for token in sentence] for sentence in doc] for doc in source_docs
    ]
    return {
        'text': source,
        'sents': source_docs,
        'sent_toks': source_doc_sents_tok
    }


class Seq2SeqCollate:
    def __init__(self, tokenizer, max_input_length=8192, max_output_length=512, add_cols=None, split=None, verbose=False):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.add_cols = [] if add_cols is None else add_cols
        self.split = split
        additional_ids = self.tokenizer.additional_special_tokens_ids
        self.special_id_min = None if len(additional_ids) == 0 else min(self.tokenizer.additional_special_tokens_ids)
        self.verbose = verbose

    def sent_extract_labels(self, batch_list, batch):
        batch_size = len(batch_list)
        num_cls_per_batch = batch['cls_mask'].sum(dim=1)
        max_num_sent = num_cls_per_batch.max()
        batch_plan_labels = [x['plan_labels'] for x in batch_list]
        doc_offset = 1  # Document CLS token comes first
        batch_plan_labels_pad = np.zeros([batch_size, max_num_sent], dtype=np.float)
        for batch_idx in range(batch_size):
            pl = batch_plan_labels[batch_idx]
            num_cls = num_cls_per_batch[batch_idx]
            for sent_idx in pl:
                try:
                    batch_plan_labels_pad[batch_idx, sent_idx + doc_offset] = 1
                except:
                    if self.verbose:
                        print(f'Sentence {sent_idx} truncated by tokenizer and cannot be assigned as part of oracle.')
            batch_plan_labels_pad[batch_idx, num_cls:] = -100  # This is padded
        batch_plan_labels_pad[:, 0] = -100  # This is the document CLS token
        batch_plan_labels_pad = torch.from_numpy(batch_plan_labels_pad)
        batch['plan_labels'] = batch_plan_labels_pad

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            [x['source'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(
                [x['target'] for x in batch_list],
                padding='longest',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors='pt'
            )

        batch = {}
        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        batch['labels'] = outputs.input_ids
        batch['cls_mask'] = batch['input_ids'] >= self.special_id_min
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100
        for col in self.add_cols:
            batch[col] = [x[col] for x in batch_list]

        if 'plan_labels' in batch_list[0] and batch_list[0]['plan_labels'] is not None:
            self.sent_extract_labels(batch_list, batch)

        if 'neg_plans' in batch_list[0]:
            neg_plans = list(itertools.chain(*[x['neg_plans'] for x in batch_list]))

            with self.tokenizer.as_target_tokenizer():
                neg_labels = self.tokenizer(
                    neg_plans,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_output_length,
                    return_tensors='pt'
                )['input_ids']

                batch['neg_labels'] = neg_labels
                # We have to make sure that the PAD token is ignored
                batch['neg_labels'][torch.where(batch['neg_labels'] == 1)] = -100
                batch['neg_labels'][torch.where(batch['neg_labels'] == 2)] = -100
        if 'pos_plans' in batch_list[0]:
            neg_plans = list(itertools.chain(*[x['pos_plans'] for x in batch_list]))

            with self.tokenizer.as_target_tokenizer():
                pos_labels = self.tokenizer(
                    neg_plans,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_output_length,
                    return_tensors='pt'
                )['input_ids']

                batch['pos_labels'] = pos_labels
                # We have to make sure that the PAD token is ignored
                batch['pos_labels'][torch.where(batch['pos_labels'] == 1)] = -100
                batch['pos_labels'][torch.where(batch['pos_labels'] == 2)] = -100
        return batch


def get_path_from_exp(weights_dir, experiment):
    dir = os.path.join(weights_dir, experiment)
    paths = list(Path(dir).rglob('*.ckpt'))
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')
