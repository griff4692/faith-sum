import os
import numpy as np
from pathlib import Path

import itertools
import nltk
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
    def __init__(self, tokenizer, max_input_length=8192, max_output_length=512, add_cols=None, split=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.add_cols = [] if add_cols is None else add_cols
        self.split = split
        self.sentence_tok_ids = self.tokenizer.additional_special_tokens_ids
        self.cutoff = min(self.tokenizer.additional_special_tokens_ids)

    def __call__(self, batch_list):
        batch_size = len(batch_list)
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
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100

        sent_pos_ids = np.zeros([batch_size, batch['input_ids'].size()[-1]], dtype=np.long)
        for batch_idx in range(batch_size):
            id_seq = batch['input_ids'][batch_idx]
            pad_idxs = torch.where(id_seq == self.tokenizer.pad_token_id)[0]

            if len(pad_idxs) == 0:
                pad_start = len(id_seq)
            else:
                pad_start = pad_idxs[0].item()
            sent_cls_pos = torch.where(id_seq >= self.cutoff)[0].tolist()
            for sent_idx, offset in enumerate(sent_cls_pos):
                start = offset
                end = sent_cls_pos[sent_idx + 1] if sent_idx < len(sent_cls_pos) - 1 else pad_start
                is_odd_sent = sent_idx % 2 != 0
                sent_pos_ids[batch_idx, start:end] = 1 if is_odd_sent else 2  # sent_idx + 1
        sent_pos_ids = torch.from_numpy(sent_pos_ids)
        for col in self.add_cols:
            batch[col] = [x[col] for x in batch_list]

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
        batch['sent_pos_ids'] = sent_pos_ids
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
