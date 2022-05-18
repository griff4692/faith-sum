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
        self.special_id_min = 999999 if len(additional_ids) == 0 else min(self.tokenizer.additional_special_tokens_ids)
        self.verbose = verbose

    def sent_extract_labels(self, batch_list, batch):
        batch_size = len(batch_list)
        num_cls = batch['cls_mask'].sum(dim=1)
        batch_plan_labels = [x['plan_labels'] for x in batch_list]
        if self.split == 'train':
            priorities = [x['sent_priority'] for x in batch_list]
            priorities_trunc = [x[:num_cls[batch_idx]] for batch_idx, x in enumerate(priorities)]
            priorities_trunc = [list(np.argsort(-x)) for x in priorities_trunc]
            batch['priority'] = priorities_trunc
        batch_plan_q = [x['plan_q'] for x in batch_list]
        batch_plan_labels_pad = np.zeros(shape=(batch_size, batch['input_ids'].size()[1]), dtype=np.float)
        batch_plan_labels_pad.fill(-100)
        valid_idxs = []
        for batch_idx in range(batch_size):
            sent_locs = batch['cls_mask'][batch_idx].nonzero().squeeze(1)
            any_pos_labels = False
            for sent_idx, location in enumerate(sent_locs):
                label = 1 if sent_idx in batch_plan_labels[batch_idx] else 0
                if label == 1:
                    any_pos_labels = True
                batch_plan_labels_pad[batch_idx, location] = label
            if any_pos_labels:
                valid_idxs.append(batch_idx)
        batch_plan_labels_pad = torch.from_numpy(batch_plan_labels_pad)
        batch['plan_labels'] = batch_plan_labels_pad
        if batch_plan_q[0] is not None:
            trunc_qs = []
            trunc_sent_nums = batch['cls_mask'].sum(dim=1).tolist()
            trunc_plan_nums = (batch_plan_labels_pad == 1).sum(dim=1).tolist()
            for sent_num, plan_n, q in zip(trunc_sent_nums, trunc_plan_nums, batch_plan_q):
                q_plan_n, q_sent_num = q.shape
                q_labels = q.argmax(axis=1)
                assert q_sent_num >= sent_num
                q = q[:, :sent_num]
                non_trunc_steps = []
                for step, label in enumerate(q_labels):
                    if label < sent_num:
                        non_trunc_steps.append(step)
                q = q[non_trunc_steps, :]
                assert plan_n == len(non_trunc_steps)
                q = np.vstack([q, np.zeros([1, q.shape[-1]])])
                q = np.hstack([q, np.zeros([q.shape[0], 1])])
                q[-1, -1] = 1  # Last token is the stop token
                trunc_qs.append(torch.from_numpy(q))

            batch['plan_q'] = trunc_qs
        return valid_idxs

    def __call__(self, batch_list):
        batch = {}
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            [x['source'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        if batch_list[0]['target'] is not None:  # If we are just doing sentence scoring, 'score', this will be None
            with self.tokenizer.as_target_tokenizer():
                outputs = self.tokenizer(
                    [x['target'] for x in batch_list],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_output_length,
                    return_tensors='pt'
                )
                batch['labels'] = outputs.input_ids
                # We have to make sure that the PAD token is ignored
                batch['labels'][torch.where(batch['labels'] == 1)] = -100

        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        batch['cls_mask'] = batch['input_ids'] >= self.special_id_min
        for col in self.add_cols:
            batch[col] = [x[col] for x in batch_list]

        if 'plan_labels' in batch_list[0] and batch_list[0]['plan_labels'] is not None:
            valid_idxs = self.sent_extract_labels(batch_list, batch)
            if len(valid_idxs) < len(batch_list) and self.split == 'train':
                num_to_remove = len(batch_list) - len(valid_idxs)
                print(f'Removing {num_to_remove} examples where the plan label has been truncated bc after 1024 WPs')
                new_batch_list = [batch_list[i] for i in valid_idxs]
                return self(new_batch_list)

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
