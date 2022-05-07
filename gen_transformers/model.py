from collections import defaultdict
import os
import regex as re

from datasets import load_metric
import pandas as pd
import pytorch_lightning as pl
from nltk import trigrams
import numpy as np
import spacy
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother

from scipy.special import expit
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr  # Necessary for tune_lr to work with PytorchLightning
        self.rouge = load_metric('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge_metric = RougeMetric()
        # <sep> separates plan from abstract for plan models
        if 'plan' in self.hparams.summary_style:
            self.special_id_cutoff = self.tokenizer.convert_tokens_to_ids('<sep>')

        self.sent_classifier, self.sent_loss = None, None
        if 'score' in self.hparams.summary_style:
            self.sent_classifier = nn.Linear(self.model.config.d_model, 1)
            pos_weight = torch.FloatTensor([1.0])
            self.sent_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def compute_ll(self, batch, labels, output):
        batch_size = len(output[1])
        num_cand = len(labels) // batch_size
        # Ignore the <eos> token
        # Tile this for each num_neg or num_pos decoder target
        model_dim = self.model.config.d_model  # 768 for BART (24 for debug shleifer/tiny-random)
        encoder_outputs = output.encoder_last_hidden_state.unsqueeze(1).repeat(
            1, num_cand, 1, 1).contiguous().view(batch_size * num_cand, -1, model_dim).clone()
        encoder_attention_mask = batch['attention_mask'].unsqueeze(1).repeat(
            1, num_cand, 1).contiguous().view(batch_size * num_cand, -1)
        inputs = {
            'encoder_outputs': [encoder_outputs],
            'attention_mask': encoder_attention_mask,
            'labels': labels,
        }

        outputs = self.model(**inputs, use_cache=False)
        # Separately compute NLL for plan (pre <sep>) and NLL for abstract (post <sep>)
        plan_nll, abstract_nll = [], []
        for cand_idx in range(len(labels)):
            label_row = labels[cand_idx]
            cutoff = int(torch.where(label_row == self.special_id_cutoff)[0])

            label_row_plan = label_row[:cutoff]
            loss_row_plan = {'logits': outputs['logits'][cand_idx, : cutoff, :]}
            plan_nll.append(self.label_smoother(loss_row_plan, label_row_plan))

            label_row_abstract = label_row[cutoff + 1:]
            loss_row_abstract = {'logits': outputs['logits'][cand_idx, cutoff + 1:, :]}
            abstract_nll.append(self.label_smoother(loss_row_abstract, label_row_abstract))

        return plan_nll, abstract_nll

    def contrast_loss(self, batch_size, num_neg_cand, batch_nll_pos, batch_nll_neg):
        """
        :param batch_size: Batch Size
        :param num_neg_cand: Number of negative targets for each example in batch (batch_size * num_neg_cand total)
        :param batch_nll_pos: Negative Log Likelihoods for generating the 'positive' ground-truth
        :param batch_nll_neg: Negative Log Likelihoods for generating the 'negative' ground-truth
        :return: Margin losses for each pair of pos and neg.  Positive targets should have a higher LL than negatives,
        with a margin defined by --contrast_margin.
        """
        margin_losses = []
        for batch_idx in range(batch_size):
            pos_ll = - batch_nll_pos[batch_idx]
            neg_nll_cands = batch_nll_neg[batch_idx * num_neg_cand: (batch_idx + 1) * num_neg_cand]
            for neg_nll_cand in neg_nll_cands:
                neg_ll_cand = - neg_nll_cand
                margin_losses.append(torch.clamp(neg_ll_cand - pos_ll + self.hparams.contrast_margin, min=0))
        return margin_losses

    def build_summaries(self, source_sents, y_hat, trigram_block=True, max_num_sents=3):
        all_summaries = []
        y_hat_cls = y_hat[np.where(y_hat > float('-inf'))]
        priority = expit(y_hat_cls.copy())
        trigram_to_sent_idx = sent_idx_to_trigram = None
        if trigram_block:
            trigram_to_sent_idx = defaultdict(list)
            sent_idx_to_trigram = defaultdict(list)

            for sent_idx, sent in enumerate(source_sents):
                sent_toks = [t for t in re.sub('\W+', ' ', sent.lower()).split(' ') if len(t) > 0]
                sent_trigrams = list(trigrams(sent_toks))
                for trigram in sent_trigrams:
                    trigram_to_sent_idx[trigram].append(sent_idx)
                    sent_idx_to_trigram[sent_idx].append(trigram)

        for k in range(min(max_num_sents, len(source_sents))):
            top_sent = np.argmax(priority)
            priority[top_sent] = float('-inf')
            if trigram_block:
                for trigram in sent_idx_to_trigram[top_sent]:
                    for other_sent_idx in trigram_to_sent_idx[trigram]:
                        priority[other_sent_idx] = float('-inf')
            # Matching Trigrams
            prev_sents = [] if k == 0 else all_summaries[k - 1]
            summary_at_k = prev_sents + [top_sent]
            all_summaries.append(summary_at_k)
        return all_summaries

    def training_step(self, batch, batch_idx):
        batch_size = len(batch['input_ids'])

        # Train on extractive labels
        plan_labels = batch.pop('plan_labels', None)
        cls_mask = batch.pop('cls_mask')

        # Perturbed Plans
        neg_labels = batch.pop('neg_labels', None)
        pos_labels = batch.pop('pos_labels', None)
        _ = batch.pop('reference', None)

        if self.hparams.summary_style != 'score':
            output = self.model(**batch, use_cache=False)
            # Regular MLE decoder loss
            loss = output.loss
            self.log('train_loss', loss, on_epoch=False, on_step=True, prog_bar=True)
            # Return label-smoothed loss
            full_loss = self.label_smoother(output, batch['labels'])
            encoder_h = output.encoder_last_hidden_state
        else:
            full_loss = 0
            output = self.model.model.encoder(**batch)
            encoder_h = output.last_hidden_state

        if plan_labels is not None:
            # Predict if a sentence is in oracle summary
            sent_preds = self.sent_classifier(encoder_h).squeeze(1)
            sent_loss = self.sent_loss(sent_preds.flatten(), plan_labels.flatten())

            # These are the document [CLS] token and padding
            # These are the document [CLS] token and padding
            sent_label_mask = (plan_labels == -100).flatten()
            sent_loss.masked_fill_(sent_label_mask, 0)
            extract_loss = sent_loss.sum() / (plan_labels != -100).sum()
            self.log('train_extract', extract_loss, on_epoch=False, on_step=True, prog_bar=True)
            if full_loss is None:
                full_loss = extract_loss
            else:
                full_loss += extract_loss
                self.log('train_combined', full_loss, on_epoch=False, on_step=True, prog_bar=True)

        if pos_labels is not None:
            # Single positive is ground-truth plan and abstractive reference
            batch_plan_nll_pos, batch_abstract_nll_pos = self.compute_ll(batch, pos_labels, output)
            # Neg labels are perturbed plans: add-1 sentence, remove-1 sentence, replace 1 sentence
            batch_plan_nll_neg, batch_abstract_nll_neg = self.compute_ll(batch, neg_labels, output)

            num_neg_cand = len(neg_labels) // batch_size

            # Extractive Plan Contrast Loss
            # Train log likelihood of generating ground-truth plan to be higher than perturbed
            plan_batch_margins = self.contrast_loss(
                batch_size, num_neg_cand, batch_plan_nll_pos, batch_plan_nll_neg
            )

            # Consistency Loss
            # Train log likelihood of generating abstract to be higher
            # when conditioned on oracle plan than when conditioned on corrupted plans (mismatched)
            abstract_batch_margins = self.contrast_loss(
                batch_size, num_neg_cand, batch_abstract_nll_pos, batch_abstract_nll_neg
            )

            # Average both and add to MLE language model loss depending on if plan and/or abstract in --contrast_modes
            plan_avg_margin = torch.stack(plan_batch_margins).mean()
            abstract_avg_margin = torch.stack(abstract_batch_margins).mean()

            # 3 modes for contrastive learning
            # 1) Just train on plan LL margin
            # 2) Just on abstract LL margin
            # 3) Both
            if 'plan' in self.hparams.contrast_modes and 'abstract' in self.hparams.contrast_modes:
                self.log('train_plan_margin', plan_avg_margin, on_epoch=False, on_step=True, prog_bar=True)
                self.log('train_abstract_margin', abstract_avg_margin, on_epoch=False, on_step=True, prog_bar=True)
                # --contrast_lambda determines the relative weight of LM versus contrast margin losses
                full_loss += self.hparams.contrast_lambda * (plan_avg_margin + abstract_avg_margin)
            elif 'plan' in self.hparams.contrast_modes:
                self.log('train_plan_margin', plan_avg_margin, on_epoch=False, on_step=True, prog_bar=True)
                # --contrast_lambda determines the relative weight of LM versus contrast margin losses
                full_loss += self.hparams.contrast_lambda * plan_avg_margin
            else:  # Must be just abstract
                assert self.hparams.contrast_modes == 'abstract'
                self.log('train_abstract_margin', abstract_avg_margin, on_epoch=False, on_step=True, prog_bar=True)
                # --contrast_lambda determines the relative weight of LM versus contrast margin losses
                full_loss += self.hparams.contrast_lambda * abstract_avg_margin
            self.log('train_combined', full_loss, on_epoch=False, on_step=True, prog_bar=True)
        return full_loss

    def validation_step(self, batch, batch_idx):
        # Train on extractive labels
        plan_labels = batch.pop('plan_labels', None)
        cls_mask = batch.pop('cls_mask')
        batch_size = len(batch['input_ids'])

        validation_kwargs = {
            'num_beams': 1,
            'num_return_sequences': 1,  # Don't over-generate for validation
            'references': batch.pop('reference'),
        }

        if self.hparams.summary_style != 'score':
            output = self.model(**batch, use_cache=False)
            # Regular MLE decoder loss
            loss = output.loss
            encoder_h = output.encoder_last_hidden_state
        else:
            loss = 0
            output = self.model.model.encoder(**batch)
            encoder_h = output.last_hidden_state

        extractive_summaries = None
        if plan_labels is not None:
            assert 'score' in self.hparams.summary_style
            # Predict if a sentence is in oracle summary
            sent_preds = self.sent_classifier(encoder_h).squeeze(-1)
            sent_loss = self.sent_loss(sent_preds.flatten(), plan_labels.flatten())

            sent_preds = self.sent_classifier(encoder_h).squeeze(-1)
            # Only predict positively for CLS tokens
            sent_preds.masked_fill_(~ cls_mask, float('-inf'))

            source = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
            source_sents = list(map(lambda x: re.split(
                r'<s\d*>', x.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip())[1:], source))
            y_hat_np = sent_preds.detach().cpu().numpy()
            extractive_summaries = []
            for batch_idx in range(batch_size):
                summary = self.build_summaries(source_sents=source_sents[batch_idx], y_hat=y_hat_np[batch_idx, :])[-1]
                extractive_summaries.append(summary)

            # These are the document [CLS] token and padding
            sent_label_mask = (plan_labels == -100).flatten()
            sent_loss.masked_fill_(sent_label_mask, 0)
            extract_loss = sent_loss.sum() / (plan_labels != -100).sum()
            self.log('val_extract', extract_loss, on_epoch=True, on_step=False, prog_bar=True)
            if loss is None:  # No generation
                loss = extract_loss
            else:
                loss += extract_loss
                self.log('val_combined', loss, on_epoch=True, on_step=False, prog_bar=True)

        all_metrics = {'val_loss': loss}

        if self.hparams.summary_style == 'score':
            gen_output_list = []
            for batch_idx in range(batch_size):
                row = self.parse_score(
                    batch['input_ids'][batch_idx].unsqueeze(0), [validation_kwargs['references'][batch_idx]],
                    extractive_summaries[batch_idx]
                )
                gen_output_list.append(row)
        else:
            gen_output_list = self.shared_generate(batch, extractive_idxs=extractive_summaries, **validation_kwargs)

        # It's a list of dictionaries --> convert into dictionary of lists and process as a batch (for ROUGE)
        gen_output = defaultdict(list)
        for item in gen_output_list:
            for k, v in item.items():
                if type(v) == list:
                    gen_output[k] += v
                elif v is not None:
                    gen_output[k].append(v)
        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        if self.hparams.summary_style in {'abstract_plan', 'plan_abstract'}:
            from_oracle_rouge = self.generate_from_oracle(batch, reduce=True, **validation_kwargs)
            all_metrics.update(from_oracle_rouge)

        if len(gen_output['abstracts']) > 0:
            all_metrics.update(self.compute_rouge(gen_output['abstracts'], gen_output['references']))
            implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
            all_metrics.update(self.compute_rouge(implied_extracts, gen_output['references'], prefix='implied_'))

        if len(gen_output['extracts']) > 0:
            all_metrics.update(self.compute_rouge(gen_output['extracts'], gen_output['references'], prefix='extract_'))

        # Measure consistency between abstract (and implied extract) and generated extract
        if len(gen_output['abstracts']) > 0 and len(gen_output['extracts']) > 0:
            all_metrics.update(self.measure_plan_abstract_consistency(batch, gen_output, **validation_kwargs))
            # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
            # If the plan is working, this should be very high (the abstract should follow the plan)
            all_metrics.update(
                self.compute_rouge(gen_output['extracts'], gen_output['abstracts'], prefix='extract_gen_')
            )

        for k, v in all_metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def score_candidates(self, reference, candidates, prefix, eval=False):
        cand_metrics = [self.compute_rouge(
            [abstract], reference, prefix=f'best_{prefix}_', eval=eval
        ) for abstract in candidates]
        best_metric = sorted(cand_metrics, key=lambda x: x[f'best_{prefix}_mean_f1'])[-1]
        return cand_metrics, best_metric

    def generate_from_oracle(self, batch, reduce=False, eval=False, **gen_kwargs):
        oracle_strs = [x.split('<sep>')[0].replace('<s>', '') for x in self.tokenizer.batch_decode(batch['labels'])]
        references = gen_kwargs.pop('references')
        results = []
        for batch_idx in range(len(batch['input_ids'])):
            decoder_input_ids = self.tokenizer.encode(
                '<s></s>' + oracle_strs[batch_idx] + '<sep>',
                add_special_tokens=False, return_tensors='pt'
            ).to(self.device)

            default_kwargs = {  # Some of these values may get overridden by gen_kwargs
                'input_ids': batch['input_ids'][batch_idx].unsqueeze(0),
                'attention_mask': batch['attention_mask'][batch_idx].unsqueeze(0),
                'decoder_input_ids': decoder_input_ids,
                'num_return_sequences': 1,
                'max_length': self.hparams.max_output_length,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
                'output_scores': True
            }

            default_kwargs.update(gen_kwargs)
            pred_ids = self.model.generate(**default_kwargs)
            pred = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            row = self.compute_rouge([pred], [references[batch_idx]], prefix='oracle_prompt_', eval=eval)
            results.append(row)
        if reduce:
            df = pd.DataFrame(results)
            return {col: df[col].dropna().mean for col in df.columns}
        return results

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        references = batch.pop('reference')
        cls_mask = batch.pop('cls_mask')
        plan_labels = batch.pop('plan_labels', None)
        use_hf_rouge = gen_kwargs.pop('use_hf_rouge')
        eval = not use_hf_rouge

        batch_size = len(references)
        gen_kwargs.update({
            'references': references
        })

        if 'score' in self.hparams.summary_style:
            assert 'score' in self.hparams.summary_style
            # Predict if a sentence is in oracle summary
            output = self.model.model.encoder(**batch)
            encoder_h = output.last_hidden_state
            sent_preds = self.sent_classifier(encoder_h).squeeze(-1)
            sent_preds.masked_fill_(~ cls_mask, float('-inf'))
            source = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
            source_sents = list(map(lambda x: re.split(
                r'<s\d*>', x.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip())[1:], source))
            y_hat_np = sent_preds.detach().cpu().numpy()
            gen_outputs = []
            for batch_idx in range(batch_size):
                summary = self.build_summaries(source_sents=source_sents[batch_idx], y_hat=y_hat_np[batch_idx, :])[-1]
                row = self.parse_score(batch['input_ids'][batch_idx].unsqueeze(0), [references[batch_idx]], summary)
                gen_outputs.append(row)
        else:
            gen_outputs = self.shared_generate(batch, **gen_kwargs)
        batch_outputs = []

        from_oracle_metrics = None
        if self.hparams.summary_style in {'abstract_plan', 'plan_abstract'}:
            from_oracle_metrics = self.generate_from_oracle(batch, use_hf_rouge, **gen_kwargs)

        for batch_idx, (reference, gen_output) in enumerate(zip(references, gen_outputs)):
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

            if from_oracle_metrics is not None:
                save_out.update(from_oracle_metrics[batch_idx])

            # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
            if gen_output['abstracts'] is not None:  # Take top of the beam or first returned sequence
                save_out.update(self.compute_rouge(gen_output['abstracts'][:1], [reference], eval=eval))
                implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
                save_out.update(self.compute_rouge(
                    implied_extracts[:1], [reference], prefix='implied_', eval=eval
                ))

                # Get all the ROUGE abstracts (average ROUGE-1, ROUGE-2)
                abstract_cand_metrics, best_abstract_metric = self.score_candidates(
                    [reference], gen_output['abstracts'], 'abstract', eval=eval
                )
                save_out.update(best_abstract_metric)
                save_out['abstract_rouges'] = ','.join([str(x['best_abstract_mean_f1']) for x in abstract_cand_metrics])

                implied_cand_metrics, best_implied_metric = self.score_candidates(
                    [reference], implied_extracts, 'implied', eval=eval
                )
                save_out.update(best_implied_metric)
                save_out['implied_extract_rouges'] = ','.join(
                    [str(x['best_implied_mean_f1']) for x in implied_cand_metrics]
                )

            if gen_output['extracts'] is not None:
                extracts = [x['summary'] for x in gen_output['extracts']]
                save_out.update(self.compute_rouge(extracts[:1], [reference], prefix='extract_', eval=eval))

                extract_cand_metrics, best_extract_metric = self.score_candidates(
                    [reference], extracts, 'extract', eval=eval
                )
                save_out.update(best_extract_metric)
                save_out['extract_rouges'] = ','.join(
                    [str(x['best_extract_mean_f1']) for x in extract_cand_metrics]
                )

            if gen_output['pyramid_extracts'] is not None:
                save_out.update(
                    self.compute_rouge(
                        gen_output['pyramid_extracts'], [reference], prefix='pyramid_extract_', eval=eval
                    )
                )

            # Measure consistency between abstract (and implied extract) and generated extract
            if gen_output['abstracts'] is not None and gen_output['extracts'] is not None:
                save_out.update(self.measure_plan_abstract_consistency(batch, gen_output, **gen_kwargs))
                # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
                # If the plan is working, this should be very high (the abstract should follow the plan)
                extracts = [x['summary'] for x in gen_output['extracts']]
                save_out.update(
                    self.compute_rouge(extracts, gen_output['abstracts'], prefix='extract_gen_', eval=eval)
                )
            batch_outputs.append(save_out)

        return batch_outputs

    def shared_generate(self, batch, extractive_idxs=None, **gen_kwargs):
        default_kwargs = {  # Some of these values may get overridden by gen_kwargs
            'input_ids': batch['input_ids'],
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
            extractive_idx = extractive_idxs[batch_idx] if extractive_idxs is not None else None
            row = self.parse_output(
                pred_ids[batch_idx], input_ids[batch_idx].unsqueeze(0), [references[batch_idx]],
                extractive_idx=extractive_idx
            )
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

    def get_summary_from_sent_idxs(self, input_ids, extractive_idxs):
        extractive_tags = [f'<s{d}>' for d in list(sorted(extractive_idxs))]
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        decoded_input = re.sub(r'<pad>|</s>|<s>', '', decoded_inputs[0])
        source_sent_tps = re.split(r'(<s\d+>)', decoded_input)
        predicted_extractive_summary = []
        for sent_idx, tp in enumerate(source_sent_tps):
            if tp.startswith('<s') and int(re.search(r'\d+', tp).group()) in extractive_idxs:
                predicted_extractive_summary.append(source_sent_tps[sent_idx + 1].strip())
        summary = postprocess_text([' '.join(predicted_extractive_summary)])[0]
        return {
            'idxs': extractive_tags,
            'summary': summary
        }

    def parse_score(self, input_ids, references, extractive_idx):
        source_raw = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
        source_docs = [list(self.nlp(x).sents) for x in source_raw]
        source_doc_sents_tok = [
            [[str(token.text).strip() for token in sentence if len(str(token.text).strip()) > 0] for sentence in doc]
            for doc in source_docs
        ]
        source = {
            'text': source_raw,
            'sents': source_docs,
            'sent_toks': source_doc_sents_tok,
        }

        extracts = [self.get_summary_from_sent_idxs(input_ids, extractive_idx)]
        return {
            'source': source,
            'abstracts': None,
            'extracts': extracts,
            'pyramid_extracts': None,
            'implied_extracts': None,
            'references': references,
        }

    def parse_output(self, pred_ids, input_ids, references, extractive_idx=None):
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
            'sent_toks': source_doc_sents_tok,
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
                decoded_input = re.sub(r'<pad>|</s>|<s>', '', decoded_input)
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
        elif self.hparams.summary_style == 'score_abstract':
            abstracts = pred_str
            assert extractive_idx is not None
            extracts = [self.get_summary_from_sent_idxs(input_ids, extractive_idx)]
        elif self.hparams.summary_style == 'abstract':
            extracts = None
            abstracts = pred_str
        elif self.hparams.summary_style == 'hybrid_control':
            source_special = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=False)
            is_extractive = ['<extract>' in p for p in source_special]
            extracts = [self.ensure_extract(
                pred_str[i], source_docs[i], source_doc_sents_tok[i]) for i in range(len(is_extractive))
                if is_extractive[i]
            ]
            abstracts = [pred_str[i] for i in range(len(is_extractive)) if not is_extractive[i]]
            # abstract_idxs = [i for i in abstract_idxs if not is_extractive[i]]
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

        pyramid_extracts = None
        # TODO Add in for regular validation
        # if eval and extracts is not None:
        #     source_sents = source['sents'][0]
        #     sent_idxs = list(itertools.chain(*[[int(y.lstrip('<s').rstrip('>')) for y in x['idxs']] for x in extracts]))
        #     # If invalid sentence is generated, we can't include it in the Pyramid
        #     sent_counts = Counter([x for x in sent_idxs if x < len(source_sents)])
        #     num_to_extract = int(round(np.mean([len(x['idxs']) for x in extracts]), 0))  # Average of generated plans
        #     pyramid_idxs = [x[0] for x in sent_counts.most_common(num_to_extract)]
        #     pyramid_summary = ' '.join([str(source_sents[pyramid_idx]).strip() for pyramid_idx in pyramid_idxs])
        #     pyramid_extracts = [pyramid_summary]

        # Extracts are represented as a dictionary of 'idxs' (indices of source sentences extracted)
        # and 'summary' (actual  text)
        return {
            'source': source,
            'abstracts': abstracts,
            'extracts': extracts,
            'pyramid_extracts': pyramid_extracts,
            'implied_extracts': implied_extracts,
            'references': references,
        }

    def measure_plan_abstract_consistency(self, batch, outputs, eval=False, **gen_kwargs):
        source_annotated = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=False)
        source_sents = [re.findall(r'<s\d+>', sa) for sa in source_annotated]

        extract_idxs = [x['idxs'] for x in outputs['extracts']]
        implied_extract_idxs = [x['idxs'] for x in outputs['implied_extracts']]

        extracts = [x['summary'] for x in outputs['extracts']]
        implied_extracts = [x['summary'] for x in outputs['implied_extracts']]

        metrics = self.compute_rouge(extracts, implied_extracts, eval=eval, prefix='extract_implied_')

        overlaps = []
        for batch_idx, (extract_idx, implied_idx) in enumerate(zip(extract_idxs, implied_extract_idxs)):
            source_sent_tags = source_sents[batch_idx]
            n = len(source_sent_tags)
            idxs = set([int(re.findall(r'\d+', tag)[0]) for tag in extract_idx])
            intersection = set(implied_idx).intersection(idxs)
            overlap_p = len(intersection) / len(idxs)
            overlap_r = len(intersection) / len(implied_idx)
            overlap_f1 = 0.0 if max(overlap_p, overlap_r) == 0 else 2 * overlap_p * overlap_r / (
                    overlap_p + overlap_r)

            row = {
                'extract_implied_sent_precision': overlap_p,
                'extract_implied_sent_recall': overlap_r,
                'extract_implied_sent_f1': overlap_f1
            }

            rand_plan = list(np.sort(list(np.random.choice(source_sent_tags, size=(min(n, 3)), replace=False))))

            decoder_input_ids = self.tokenizer(
                '<s></s>' + ''.join(rand_plan) + '<sep>', add_special_tokens=False, return_tensors='pt'
            )['input_ids'].to(self.device)

            default_kwargs = {  # Some of these values may get overridden by gen_kwargs
                'input_ids': batch['input_ids'][batch_idx].unsqueeze(0),
                'attention_mask': batch['attention_mask'][batch_idx].unsqueeze(0),
                'decoder_input_ids': decoder_input_ids,
                'num_return_sequences': 1,
                'max_length': self.hparams.max_output_length,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
                'output_scores': True
            }

            default_kwargs.update(gen_kwargs)
            default_kwargs.pop('references', None)
            pred_ids = self.model.generate(**default_kwargs)
            pred = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]

            pred_sents = list(self.nlp(pred).sents)
            from_rand_abstract_sents_tok = [[str(token.text) for token in sentence] for sentence in pred_sents]

            source_toks = outputs['source'][batch_idx]['sent_toks'][0]
            from_rand_implied_idx = gain_selection(
                source_toks, from_rand_abstract_sents_tok, 5, lower=True, sort=True
            )[0]

            rand_plan_int = [int(x.lstrip('<s').rstrip('>')) for x in rand_plan]
            rand_intersection = set(rand_plan_int).intersection(set(from_rand_implied_idx))
            rand_overlap_p = len(rand_intersection) / len(rand_plan_int)
            rand_overlap_r = len(rand_intersection) / len(from_rand_implied_idx)
            rand_overlap_f1 = 0.0 if max(rand_overlap_p, rand_overlap_r) == 0 \
                else 2 * rand_overlap_p * rand_overlap_r / (rand_overlap_p + rand_overlap_r)

            rand_row = {
                'rand_plan_implied_sent_precision': rand_overlap_p,
                'rand_plan_implied_sent_recall': rand_overlap_r,
                'rand_plan_implied_sent_f1': rand_overlap_f1
            }

            row.update(rand_row)
            overlaps.append(row)
        df = pd.DataFrame(overlaps)
        avgs = {k: df[k].mean() for k in df.columns}
        metrics.update(avgs)

        # Sample 1 random plan and measure consistency
        return avgs

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
