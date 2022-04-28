from collections import defaultdict
import os
import regex as re

from datasets import load_metric
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import spacy
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, BartForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother

from preprocess.convert_abstractive_to_extractive import gain_selection
from eval.rouge_metric import RougeMetric
from gen_transformers.data_utils import postprocess_text


os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = BartForConditionalGeneration.from_pretrained(hf_model)
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr  # Necessary for tune_lr to work with PytorchLightning
        self.rouge = load_metric('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge_metric = RougeMetric()

        self.sent_classifier, self.sent_loss = None, None
        if self.hparams.mode in {'all', 'extract'}:
            self.sent_classifier = nn.Linear(self.model.config.d_model, 1)
            self.sent_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def sent_classification(self, cls_mask, encoder_hidden_state):
        batch_size = len(cls_mask)
        model_dim = self.model.config.d_model
        # Predict Sentences
        sents_per_ex = cls_mask.sum(dim=1)
        max_sents_per_ex = sents_per_ex.max()
        cls_h = torch.zeros(size=(batch_size, max_sents_per_ex, model_dim), device=self.device)
        cls_h_flat = encoder_hidden_state[cls_mask.nonzero(as_tuple=True)]
        offset = 0
        for batch_idx, sent_len in enumerate(sents_per_ex):
            cls_h[batch_idx, :sent_len, :] = cls_h_flat[offset: offset + sent_len, :]

        sent_preds = self.sent_classifier(cls_h).squeeze(-1)
        return sent_preds

    def shared_step(self, batch):
        plan_labels = batch.pop('plan_labels')
        cls_mask = batch.pop('cls_mask')

        # Process with BART
        output = self.model(**batch, use_cache=False)
        # Regular MLE decoder loss
        mle_abstract_loss = output.loss
        combined_loss = self.label_smoother(output, batch['labels'])

        extract_loss = None
        if self.hparams.mode in {'all', 'extract'}:
            # Predict if a sentence is in oracle summary
            encoder_h = output.encoder_last_hidden_state
            sent_preds = self.sent_classification(cls_mask, encoder_h)
            sent_loss = self.sent_loss(sent_preds, plan_labels)

            # These are the document [CLS] token and padding
            sent_label_mask = plan_labels == -100
            sent_loss.masked_fill_(sent_label_mask, 0)
            extract_loss = (sent_loss.sum(dim=1) / (plan_labels != -100).sum(dim=1)).mean()
            combined_loss += extract_loss

        return {
            'loss': combined_loss,
            'abstract': mle_abstract_loss,
            'extract': extract_loss,
        }

    def training_step(self, batch, batch_idx):
        _ = batch.pop('reference', None)

        metrics = self.shared_step(batch)

        for k, v in metrics.items():
            if v is not None:
                self.log(f'train_{k}', v, on_epoch=False, on_step=True, prog_bar=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        validation_kwargs = {
            'num_beams': 1,
            'num_return_sequences': 1,  # Don't over-generate for validation
            'references': batch.pop('reference'),
        }
        # Process with BART
        metrics = self.shared_step(batch)
        all_metrics = {f'val_{k}': v for k, v in metrics.items() if v is not None}

        gen_output_list = self.shared_generate(batch, **validation_kwargs)

        # It's a list of dictionaries --> convert into dictionary of lists and process as a batch (for ROUGE)
        gen_output = defaultdict(list)
        for item in gen_output_list:
            for k, v in item.items():
                if type(v) == list:
                    gen_output[k] += v
                elif v is not None:
                    gen_output[k].append(v)
        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        if len(gen_output['abstracts']) > 0:
            all_metrics.update(self.compute_rouge(gen_output['abstracts'], gen_output['references']))
            implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
            all_metrics.update(self.compute_rouge(implied_extracts, gen_output['references'], prefix='implied_'))

        if len(gen_output['extracts']) > 0:
            all_metrics.update(self.compute_rouge(gen_output['extracts'], gen_output['references'], prefix='extract_'))

        # Measure consistency between abstract (and implied extract) and generated extract
        if len(gen_output['abstracts']) > 0 and len(gen_output['extracts']) > 0:
            all_metrics.update(self.measure_plan_abstract_consistency(gen_output))
            # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
            # If the plan is working, this should be very high (the abstract should follow the plan)
            all_metrics.update(
                self.compute_rouge(gen_output['extracts'], gen_output['abstracts'], prefix='extract_gen_')
            )

        for k, v in all_metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)

        return all_metrics['val_loss']

    def score_candidates(self, reference, candidates, prefix, eval=True):
        cand_metrics = [self.compute_rouge(
            [abstract], reference, prefix=f'best_{prefix}_', eval=eval
        ) for abstract in candidates]
        best_metric = sorted(cand_metrics, key=lambda x: x[f'best_{prefix}_mean_f1'])[-1]
        return cand_metrics, best_metric

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        references = batch.pop('reference')
        use_hf_rouge = gen_kwargs.pop('use_hf_rouge')
        gen_kwargs.update({
            'references': references
        })

        gen_outputs = self.shared_generate(batch, **gen_kwargs)

        batch_outputs = []
        for reference, gen_output in zip(references, gen_outputs):
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
                'reference': reference, 'source': gen_output['source']['text'][0],
                'extract_idx': extract_idx_flat, 'implied_extract_idx': implied_extract_idx_flat,
            }

            # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
            if gen_output['abstracts'] is not None:  # Take top of the beam or first returned sequence
                save_out.update(self.compute_rouge(gen_output['abstracts'][:1], [reference], eval=not use_hf_rouge))
                implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
                save_out.update(self.compute_rouge(
                    implied_extracts[:1], [reference], prefix='implied_', eval=not use_hf_rouge
                ))

                # Get all the ROUGE abstracts (average ROUGE-1, ROUGE-2)
                abstract_cand_metrics, best_abstract_metric = self.score_candidates(
                    [reference], gen_output['abstracts'], 'abstract', eval=not use_hf_rouge
                )
                save_out.update(best_abstract_metric)
                save_out['abstract_rouges'] = ','.join([str(x['best_abstract_mean_f1']) for x in abstract_cand_metrics])

                implied_cand_metrics, best_implied_metric = self.score_candidates(
                    [reference], implied_extracts, 'implied', eval=not use_hf_rouge
                )
                save_out.update(best_implied_metric)
                save_out['implied_extract_rouges'] = ','.join(
                    [str(x['best_implied_mean_f1']) for x in implied_cand_metrics]
                )

            if gen_output['extracts'] is not None:
                extracts = [x['summary'] for x in gen_output['extracts']]
                save_out.update(self.compute_rouge(extracts[:1], [reference], prefix='extract_', eval=not use_hf_rouge))

                extract_cand_metrics, best_extract_metric = self.score_candidates(
                    [reference], extracts, 'extract', eval=not use_hf_rouge
                )
                save_out.update(best_extract_metric)
                save_out['extract_rouges'] = ','.join(
                    [str(x['best_extract_mean_f1']) for x in extract_cand_metrics]
                )

                # cand_metrics = [self.compute_rouge(
                #     [extract], reference, prefix='best_extract_', eval=not hf_rouge
                # ) for extract in extracts]
                # best_metric = sorted(cand_metrics, key=lambda x: x['best_extract_mean_f1'])[-1]
                # save_out.update(best_metric)
                # save_out['extract_rouges'] = ','.join([str(x['best_extract_mean_f1']) for x in cand_metrics])

            if gen_output['pyramid_extracts'] is not None:
                save_out.update(
                    self.compute_rouge(
                        gen_output['pyramid_extracts'], [reference], prefix='pyramid_extract_', eval=not use_hf_rouge
                    )
                )

            # Measure consistency between abstract (and implied extract) and generated extract
            if gen_output['abstracts'] is not None and gen_output['extracts'] is not None:
                save_out.update(self.measure_plan_abstract_consistency(gen_output))
                # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
                # If the plan is working, this should be very high (the abstract should follow the plan)
                extracts = [x['summary'] for x in gen_output['extracts']]
                save_out.update(
                    self.compute_rouge(extracts, gen_output['abstracts'], prefix='extract_gen_', eval=eval)
                )
            batch_outputs.append(save_out)

        return batch_outputs

    def shared_generate(self, batch, **gen_kwargs):
        default_kwargs = {  # Some of these values may get overridden by gen_kwargs
            'input_ids': batch['input_ids'],
            # 'sent_pos_ids': batch['sent_pos_ids'],
            'attention_mask': batch['attention_mask'],
            'num_return_sequences': 1,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'output_scores': True
        }

        references = gen_kwargs.pop('references')
        default_kwargs.update(gen_kwargs)
        pred_ids = self.model.generate(**default_kwargs)
        gold_ids = batch['labels']
        gold_ids[torch.where(batch['labels'] == -100)] = 1
        input_ids = batch['input_ids']

        batch_size = len(input_ids)
        num_pred = len(pred_ids)
        num_cands = gen_kwargs['num_return_sequences']
        assert num_cands * batch_size == num_pred
        if num_cands > 1:
            pred_ids = pred_ids.view(batch_size, num_cands, -1)
        else:
            pred_ids = pred_ids.unsqueeze(1)
        outputs = []
        for batch_idx in range(batch_size):
            row = self.parse_output(pred_ids[batch_idx], input_ids[batch_idx].unsqueeze(0), [references[batch_idx]])
            outputs.append(row)
        return outputs

    def ensure_extract(self, pred_str, source_sents, source_sent_toks):
        extractive = []
        idxs = []
        pred_sents = list(self.nlp(pred_str).sents)
        for sent in pred_sents:
            sent_toks = [str(token.text).strip() for token in sent if len(str(token.text).strip()) > 0]
            if len(sent_toks) <= 1:
                continue
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

    def parse_output(self, pred_ids, input_ids, references):
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

        extracts = None
        abstracts = pred_str

        implied_extracts = None
        if abstracts is not None:
            implied_extracts = []
            abstract_sents = [list(self.nlp(x).sents) for x in abstracts]
            abstract_sents_tok = [[[
                str(token.text) for token in sentence] for sentence in abstract_sent] for abstract_sent in
                abstract_sents]

            if batch_size == len(abstract_sents_tok):
                for idx in range(batch_size):
                    source_toks = source['sent_toks'][idx]
                    source_raw = source['sents'][idx]
                    implied_oracle_idx = gain_selection(source_toks, abstract_sents_tok[idx], 5, lower=True, sort=True)[0]
                    implied_oracle = ' '.join([str(source_raw[i]) for i in implied_oracle_idx])
                    implied_extracts.append({
                        'idxs': implied_oracle_idx,
                        'summary': implied_oracle,
                    })
            else:
                assert len(source['sent_toks']) == 1
                for idx in range(len(abstract_sents_tok)):
                    source_toks = source['sent_toks'][0]
                    source_raw = source['sents'][0]
                    implied_oracle_idx = gain_selection(source_toks, abstract_sents_tok[idx], 5, lower=True, sort=True)[0]
                    implied_oracle = ' '.join([str(source_raw[i]) for i in implied_oracle_idx])
                    implied_extracts.append({
                        'idxs': implied_oracle_idx,
                        'summary': implied_oracle,
                    })

        # Extracts are represented as a dictionary of 'idxs' (indices of source sentences extracted)
        # and 'summary' (actual  text)
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

    def compute_rouge(self, generated, gold, prefix='', eval=False):
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        if eval:  # Use SummEval PERL script
            outputs = self.rouge_metric.evaluate_batch(generated, gold, aggregate=True)['rouge']
            f1s = []
            stats = {}
            for rouge_type in ['1', '2', 'L']:
                fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
                stats[f'{prefix}rouge{rouge_type}_precision'] = outputs[f'rouge_{rouge_type.lower()}_precision']
                stats[f'{prefix}rouge{rouge_type}_recall'] = outputs[f'rouge_{rouge_type.lower()}_recall']
                stats[f'{prefix}rouge{rouge_type}_f1'] = fscore
                f1s.append(fscore)
        else:
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps)

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
