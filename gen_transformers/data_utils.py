import os
from pathlib import Path

import nltk
import numpy as np
import torch


def postprocess_text(texts):
    return ['\n'.join(nltk.sent_tokenize(text.strip())) for text in texts]


class Seq2SeqCollate:
    def __init__(self, tokenizer, max_input_length=8192, max_output_length=512, split=None, verbose=False):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.split = split
        additional_ids = self.tokenizer.additional_special_tokens_ids
        self.special_id_min = 999999 if len(additional_ids) == 0 else min(self.tokenizer.additional_special_tokens_ids)
        self.verbose = verbose

    def sent_extract_labels(self, batch_list, batch):
        batch_size = len(batch_list)
        num_cls = batch['cls_mask'].sum(dim=1)
        labels = [x['plan_labels'] for x in batch_list]
        priorities = [x['sent_priority'] for x in batch_list]
        priorities_trunc = [x[:num_cls[batch_idx]] for batch_idx, x in enumerate(priorities)]
        priorities_trunc = [list(np.argsort(-x)) for x in priorities_trunc]
        batch['priority'] = priorities_trunc
        valid_idxs = []
        labels_trunc = []
        for batch_idx in range(batch_size):
            valid_labels = [i for i in labels[batch_idx] if i < num_cls[batch_idx]]
            if len(valid_labels) > 0:
                labels_trunc.append(torch.LongTensor(valid_labels))
                valid_idxs.append(batch_idx)
            else:
                # Dummy Label is first sentence (for validation loss)
                # During training, we filter out these examples
                labels_trunc.append(torch.LongTensor([0]))
        batch['plan_labels'] = labels_trunc
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
        batch['references'] = [x['reference'] for x in batch_list]

        if 'plan_labels' in batch_list[0] and batch_list[0]['plan_labels'] is not None:
            valid_idxs = self.sent_extract_labels(batch_list, batch)
            if len(valid_idxs) < len(batch_list) and self.split == 'train':
                num_to_remove = len(batch_list) - len(valid_idxs)
                if self.verbose:
                    print(
                        f'Removing {num_to_remove} examples where the plan label has been truncated bc after 1024 WPs'
                    )
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
