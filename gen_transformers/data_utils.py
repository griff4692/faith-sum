import itertools
import torch


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
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100

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
        return batch
