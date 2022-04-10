import regex as re

from datasets import load_metric
import pandas as pd
import pytorch_lightning as pl
import nltk
import numpy as np
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother
from convert_abstractive_to_extractive import gain_selection


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


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr
        self.train_size = None
        self.rouge = load_metric('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)
        self.nlp = spacy.load('en_core_web_sm')

        # Pull out from regular NLL
        # self.special_id_cutoff = min(self.tokenizer.additional_special_tokens_ids)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Summarization Finetuning')
        parser.add_argument('--lr', type=float, default=2.2e-4)
        parser.add_argument('--target_batch_size', type=int, default=16)
        parser.add_argument('--per_device_train_batch_size', type=int, default=8)
        parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
        parser.add_argument('--weight_decay', type=float, default=5e-5)
        parser.add_argument('--max_output_length', type=int, default=256)
        parser.add_argument('--max_input_length', type=int, default=1024)
        parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
            'facebook/bart-base',
            'facebook/bart-large',
            'Yale-LILY/brio-cnndm-uncased',
        ])
        return parent_parser

    def training_step(self, batch, batch_idx):
        output = self.model(**batch, use_cache=False)
        loss = output.loss
        self.log('train_loss', loss, on_epoch=False, on_step=True, prog_bar=True)

        smooth_loss = self.label_smoother(output, batch['labels'])
        return smooth_loss

        # TODO workshop ~ didn't seem to be working well
        # lm_mask = batch['labels'].le(self.special_id_cutoff - 1)
        # plan_mask = batch['labels'].ge(self.special_id_cutoff)
        # lm_loss = self.label_smoother(output, batch['labels'].masked_fill(plan_mask, -100))
        # plan_loss = self.label_smoother(output, batch['labels'].masked_fill(lm_mask, -100))
        # self.log('plan_loss', plan_loss, on_epoch=False, on_step=True, prog_bar=True)
        # self.log('lm_loss', lm_loss, on_epoch=False, on_step=True, prog_bar=True)
        # joint_loss = lm_loss + self.hparams.plan_lambda * plan_loss
        # return joint_loss

    def validation_step(self, batch, batch_idx):
        validation_kwargs = {
            'num_beams': 1,
            'num_return_sequences': 1,  # Don't over-generate for validation
            'references': batch.pop('reference', None),
        }
        output = self.model(**batch)
        loss = output.loss

        all_metrics = {'val_loss': loss}
        gen_output = self.shared_generate(batch, **validation_kwargs)
        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        if gen_output['abstracts'] is not None:
            all_metrics.update(self.rouge_metrics(gen_output['abstracts'], gen_output['references']))
            implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
            all_metrics.update(self.rouge_metrics(implied_extracts, gen_output['references'], prefix='implied_'))

        if gen_output['extracts'] is not None:
            all_metrics.update(self.rouge_metrics(gen_output['extracts'], gen_output['references'], prefix='extract_'))

        # Measure consistency between abstract (and implied extract) and generated extract
        if gen_output['abstracts'] is not None and gen_output['extracts'] is not None:
            all_metrics.update(self.measure_plan_abstract_consistency(gen_output))
            # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
            # If the plan is working, this should be very high (the abstract should follow the plan)
            all_metrics.update(
                self.rouge_metrics(gen_output['extracts'], gen_output['abstracts'], prefix='extract_gen_')
            )

        for k, v in all_metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        gen_kwargs.update({
            'references': batch.pop('reference', None),
        })
        output = self.model(**batch)
        loss = output.loss
        gen_output = self.shared_generate(batch, **gen_kwargs)
        reference = gen_output['references']
        assert len(reference) == 1  # TODO - add support for multi-batch outputs

        abstract_flat = '' if gen_output['abstracts'] is None else '<cand>'.join(gen_output['abstracts'])
        extract_flat = '' if gen_output['extracts'] is None else '<cand>'.join(
            [x['summary'] for x in gen_output['extracts']])
        extract_idx_flat = '' if gen_output['extracts'] is None else '<cand>'.join(
            [','.join(map(str, x['idxs'])) for x in gen_output['extracts']])
        implied_extract_flat = '' if gen_output['implied_extracts'] is None else '<cand>'.join(
            [x['summary'] for x in gen_output['implied_extracts']])
        implied_extract_idx_flat = '' if gen_output['implied_extracts'] is None else '<cand>'.join(
            [','.join(map(str, x['idxs'])) for x in gen_output['implied_extracts']])

        save_out = {
            'abstract': abstract_flat, 'extract': extract_flat, 'implied_extract': implied_extract_flat,
            'reference': reference[0], 'source': gen_output['source']['text'][0],
            'extract_idx': extract_idx_flat, 'implied_extract_idx': implied_extract_idx_flat, 'loss': loss,
        }

        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        if gen_output['abstracts'] is not None:  # Take top of the beam or first returned sequence
            save_out.update(self.rouge_metrics(gen_output['abstracts'][:1], gen_output['references'][:1]))
            implied_extracts = [x['summary'] for x in gen_output['implied_extracts'][:1]]
            save_out.update(self.rouge_metrics(implied_extracts, reference, prefix='implied_'))

        if gen_output['extracts'] is not None:
            extracts = [x['summary'] for x in gen_output['extracts']]
            save_out.update(self.rouge_metrics(extracts[:1], reference, prefix='extract_'))
            cand_metrics = [self.rouge_metrics([extract], reference, prefix='best_extract_') for extract in extracts]
            best_metric = sorted(cand_metrics, key=lambda x: x['best_extract_mean_f1'])[-1]
            save_out.update(best_metric)

        # Measure consistency between abstract (and implied extract) and generated extract
        if gen_output['abstracts'] is not None and gen_output['extracts'] is not None:
            save_out.update(self.measure_plan_abstract_consistency(gen_output))
            # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
            # If the plan is working, this should be very high (the abstract should follow the plan)
            save_out.update(
                self.rouge_metrics(gen_output['extracts'], gen_output['abstracts'], prefix='extract_gen_')
            )

        return save_out

    def shared_generate(self, batch, **gen_kwargs):
        default_kwargs = {  # These may get overridden by gen_kwargs
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_return_sequences': 1,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'output_scores': True
        }
        references = gen_kwargs.pop('references', None)
        default_kwargs.update(gen_kwargs)
        pred_ids = self.model.generate(**default_kwargs)
        gold_ids = batch['labels']
        gold_ids[torch.where(batch['labels'] == -100)] = 1
        input_ids = batch['input_ids']
        outputs = self.parse_output(pred_ids, gold_ids, input_ids, references=references)
        return outputs

    def ensure_extract(self, pred_str, source_sents, source_sent_toks):
        extractive = []
        idxs = []
        for sent in list(self.nlp(pred_str).sents):
            sent_toks = [str(token.text).strip() for token in sent if len(str(token.text).strip()) > 0]
            max_intersect = 0
            closest_sent = ''
            best_idx = -1
            for source_idx in range(len(source_sents)):
                num_intersect = len(set(source_sent_toks[source_idx]).intersection(set(sent_toks)))
                if num_intersect >= max_intersect:
                    closest_sent = source_sents[source_idx]
                    max_intersect = num_intersect
                    best_idx = source_idx
            idxs.append(best_idx)
            extractive.append(str(closest_sent))
        return {'summary': ' '.join(extractive), 'idxs': idxs}

    def parse_output(self, pred_ids, gold_ids, input_ids, references=None):
        if references is None:
            references = self.tokenizer.batch_decode(gold_ids.tolist(), skip_special_tokens=True)
        pred_str = self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        batch_size = len(input_ids)

        source_raw = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
        source_docs = [list(self.nlp(x).sents) for x in source_raw]
        source_doc_sents_tok = [
            [[str(token.text).strip() for token in sentence if len(str(token.text).strip()) > 0] for sentence in doc]
            for doc in source_docs
        ]
        source = {
            'text': source_raw,
            'sents': source_docs,
            'sent_toks': source_doc_sents_tok
        }

        if 'plan' in self.hparams.summary_style:
            decoded_sent_preds = [
                re.findall(r'(<s\d+>)', x) for x in
                self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=False)
            ]
            decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decoded_inputs = list(np.repeat(decoded_inputs, len(decoded_sent_preds) // len(decoded_inputs)))
            assert len(decoded_inputs) == len(decoded_sent_preds)
            extracts = []  # Get them from the plan
            for example_idx, decoded_input in enumerate(decoded_inputs):
                source_sent_tps = re.split(r'(<s\d+>)', decoded_input)
                first_sent = source_sent_tps[2]  # normally, empty space, '<s0>, {body first sentence}
                predicted_extractive_summary = []
                for sent_idx, tp in enumerate(source_sent_tps):
                    if tp in decoded_sent_preds[example_idx]:
                        predicted_extractive_summary.append(source_sent_tps[sent_idx + 1].strip())
                #  (should only happen before training starts)
                if len(predicted_extractive_summary) == 0:  # Default to LEAD-1 if none predicted
                    assert '<s' not in first_sent
                    extracts.append({'idxs': ['<s0>'], 'summary': first_sent})
                else:
                    extracts.append({
                        'idxs': decoded_sent_preds[example_idx],
                        'summary': postprocess_text([' '.join(predicted_extractive_summary)])[0]
                    })
            abstracts = None if self.hparams.summary_style == 'plan' else pred_str
        elif self.hparams.summary_style == 'extract':  # Our abstractive generation is actually an extract
            extracts = list(map(
                lambda i: self.ensure_extract(pred_str[i], source_docs[i], source_doc_sents_tok[i]), range(batch_size)
            ))
            abstracts = None
        elif self.hparams.summary_style == 'abstract':
            extracts = None
            abstracts = pred_str
        elif self.hparams.summary_style == 'hybrid_control':
            preds = self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=False)
            is_extractive = ['<extract>' in p for p in preds]
            extracts = [preds[i] for i in range(len(is_extractive)) if is_extractive[i]]
            abstracts = [preds[i] for i in range(len(is_extractive)) if not is_extractive[i]]
            if len(extracts) == 0:
                extracts = None
            if len(abstracts) == 0:
                abstracts = None
        else:
            raise Exception(f'Unrecognized summary style -> {self.hparams.summary_style}')

        implied_extracts = None
        if abstracts is not None:
            implied_extracts = []
            abstract_sents = [list(self.nlp(x).sents) for x in abstracts]
            abstract_sents_tok = [[[
                str(token.text) for token in sentence] for sentence in abstract_sent] for abstract_sent in
                abstract_sents]
            for batch_idx in range(batch_size):
                source_toks = source['sent_toks'][batch_idx]
                implied_oracle_idx = gain_selection(
                    source_toks, abstract_sents_tok[batch_idx], 5, lower=True, sort=True
                )[0]
                implied_oracle = ' '.join([str(source_toks[i]) for i in implied_oracle_idx])
                implied_extracts.append({
                    'idxs': implied_oracle_idx,
                    'summary': implied_oracle,
                })

        # Extracts are represented as a dictionary of 'idxs' (indices of source sentences extracted)
        # and 'summary' (actual text)
        return {
            'source': source,
            'abstracts': abstracts,
            'extracts': extracts,
            'implied_extracts': implied_extracts,
            'references': references,
        }

    def measure_plan_abstract_consistency(self, outputs):
        extract_idxs = [x['idxs'] for x in outputs['extracts']]
        implied_extract_idxs = [x['idxs'] for x in outputs['implied_extracts']]
        overlaps = []
        for extract_idx, implied_idx in zip(extract_idxs, implied_extract_idxs):
            idxs = set([int(re.findall(r'\d+', tag)[0]) for tag in extract_idx])
            intersection = set(implied_idx).intersection(idxs)
            overlap_p = len(intersection) / len(idxs)
            overlap_r = len(intersection) / len(implied_idx)
            overlap_f1 = 0.0 if max(overlap_p, overlap_r) == 0 else 2 * overlap_p * overlap_r / (
                    overlap_p + overlap_r)
            overlaps.append({
                'extract_implied_sent_precision': overlap_p,
                'extract_implied_sent_recall': overlap_r,
                'extract_implied_sent_f1': overlap_f1
            })
        df = pd.DataFrame(overlaps)
        return {k: df[k].mean() for k in df.columns}

    def rouge_metrics(self, generated, gold, prefix=''):
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        rouge_output = self.rouge.compute(predictions=generated, references=gold, rouge_types=rouge_types)
        stats = {}
        f1s = []
        for rouge_type in rouge_types:
            stats[f'{prefix}{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
            stats[f'{prefix}{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
            stats[f'{prefix}{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
            f1s.append(rouge_output[rouge_type].mid.fmeasure)
        stats[f'{prefix}mean_f1'] = np.array(f1s).mean()
        return stats

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        nps = list(self.named_parameters())
        grouped_parameters = [
            {
                'params': [p for n, p in nps if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in nps if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        if self.hparams.no_schedule or self.hparams.debug or self.hparams.find_lr:
            return optimizer

        # 6% is somewhat standard for fine-tuning Transformers (can be a tunable hyper-parameter as well)
        # nonzero warmup helps mitigate risk of catastrophic forgetting from pre-training (big risk bc/ of new domain)
        # warmup = round(0.06 * self.hparams.max_steps)
        warmup = 200
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=self.hparams.max_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
