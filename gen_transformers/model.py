from collections import defaultdict
import os
import regex as re

from datasets import load_metric
import pandas as pd
import pytorch_lightning as pl
from nltk import trigrams
import numpy as np
import spacy
from scipy.special import expit
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother
from transformers.models.bart.modeling_bart import _expand_mask, BartEncoderLayer

from preprocess.extract_oracles import convert_to_sents
from preprocess.convert_abstractive_to_extractive import gain_selection
from eval.rouge_metric import RougeMetric


os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


def sentence_mask(cls_mask, sent_idx_to_mask):
    sent_mask = torch.zeros_like(cls_mask, device=cls_mask.device).long()
    sent_locs = cls_mask.nonzero()[:, 1]
    num_sents = len(sent_locs)
    for sent_idx, sent_loc in enumerate(sent_locs):
        sent_loc = sent_loc.item()
        end_loc = sent_locs[sent_idx + 1].item() if sent_idx + 1 < num_sents else len(sent_locs)
        if sent_idx in sent_idx_to_mask:
            sent_mask[0, sent_loc:end_loc] = 1
    return sent_mask


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.model.config.output_attentions = True  # Keep track of them
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
            self.sent_model = None
            # if self.hparams.sent_model_layer:
            #     self.sent_model = BartEncoderLayer(self.model.config)
            #     self.sent_model.load_state_dict(self.model.model.encoder.layers[-1].state_dict())
            self.kld_loss = nn.KLDivLoss(log_target=False, reduction='batchmean')
            self.enc_transform = nn.Linear(self.model.config.d_model, 200, bias=True)
        self.sim = nn.CosineSimilarity()

    def parse_source_text_from_inputs(self, batch):
        source_special = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        source_no_special = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=True)
        if self.hparams.add_sent_toks:  # If we have sentences demarcated in the source text
            source_sents = list(map(lambda x: re.split(
                r'<s\d*>', x.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip())[1:], source_special))
            source_sents = list(map(lambda ss: list(map(lambda x: x.strip(), ss)), source_sents))
            source_sent_toks = [
                [[str(t.text).strip() for t in self.nlp(sentence) if len(str(t.text).strip()) > 0] for sentence in doc]
                for doc in source_sents
            ]
        else:  # We have to sentence split ourselves
            source_sents = [convert_to_sents(x, self.nlp) for x in source_no_special]
            source_sent_toks = [
                [[str(t.text).strip() for t in sentence if len(str(t.text).strip()) > 0] for sentence in doc]
                for doc in source_sents
            ]
            source_sents = list(map(lambda ss: [str(x).strip() for x in ss], source_sents))
        return [{'text': a.strip(), 'sents': b, 'sent_toks': c} for a, b, c in
                zip(source_no_special, source_sents, source_sent_toks)]

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

    def build_summaries(self, source, y_hat, trigram_block=True, max_num_sents=3):
        all_summaries = []
        y_hat_cls = y_hat[np.where(y_hat > float('-inf'))]
        priority = expit(y_hat_cls.copy())
        trigram_to_sent_idx = sent_idx_to_trigram = None
        if trigram_block:
            trigram_to_sent_idx = defaultdict(list)
            sent_idx_to_trigram = defaultdict(list)

            for sent_idx, sent in enumerate(source['sents']):
                sent_toks = [t for t in re.sub('\W+', ' ', sent.lower()).split(' ') if len(t) > 0]
                sent_trigrams = list(trigrams(sent_toks))
                for trigram in sent_trigrams:
                    trigram_to_sent_idx[trigram].append(sent_idx)
                    sent_idx_to_trigram[sent_idx].append(trigram)

        for k in range(min(max_num_sents, len(source['sents']))):
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
        summary_idx = all_summaries[-1]
        return_obj = self.get_summary_from_sent_idxs(source, summary_idx)
        return_obj['sent_dist'] = y_hat_cls
        return return_obj

    def score_sents(self, cls_mask, encoder_hidden_states):
        if self.sent_model is not None:
            cls_attention_mask = cls_mask.float()
            cls_attention_mask[:, 0] = 1.0  # Document BOS should be un-masked
            cls_attention_mask_exp = _expand_mask(cls_attention_mask, dtype=encoder_hidden_states.dtype)
            encoder_hidden_states = self.sent_model(
                hidden_states=encoder_hidden_states, attention_mask=cls_attention_mask_exp, layer_head_mask=None
            )[0]
        return self.sent_classifier(encoder_hidden_states).squeeze(-1)

    def kld(self, y_hat_scores, y_dist):
        y_hat_lprob = torch.log_softmax(y_hat_scores, dim=-1)
        return self.kld_loss(y_hat_lprob, y_dist.float())

    def compute_ll_extract_loss(self, sent_preds, plan_labels):
        sent_loss = self.sent_loss(sent_preds.flatten(), plan_labels.flatten())

        # These are the document [CLS] token and padding
        sent_label_mask = (plan_labels == -100).flatten()
        sent_loss.masked_fill_(sent_label_mask, 0)
        extract_loss = sent_loss.sum() / (plan_labels != -100).sum()
        return extract_loss

    def compute_kld_extract_loss(self, sent_preds, plan_q, cls_mask):
        batch_size = len(sent_preds)
        klds = []
        for batch_idx in range(batch_size):
            p, q = sent_preds[batch_idx, cls_mask[batch_idx]], plan_q[batch_idx, cls_mask[batch_idx]]
            kld_loss = self.kld(p.unsqueeze(0), q.unsqueeze(0))
            klds.append(kld_loss)
        extract_loss = torch.stack(klds).mean()
        return extract_loss

    def compute_correlation(self, sent_preds, plan_q, cls_mask):
        batch_size = len(sent_preds)
        corels = []
        for batch_idx in range(batch_size):
            p = sent_preds[batch_idx, cls_mask[batch_idx]].detach().cpu().numpy()
            q = plan_q[batch_idx, cls_mask[batch_idx]].detach().cpu().numpy()
            try:
                corel = pearsonr(p, q)[0]
            except ValueError:
                print('Invalid Correlation Inputs.  Skipping.')
                continue
            corels.append(corel)
        return np.mean(corels)

    def brio_step(self, priority, cls_mask, batch, reference, encoder_h, mask_upperbound=10):
        batch_size = len(priority)
        margin_losses = []
        pos_neg_gap = []
        for batch_idx in range(batch_size):
            p = priority[batch_idx]
            input_ids = batch['input_ids'][batch_idx]
            mask_range = list(range(1, mask_upperbound + 1))
            masks = []
            for num in mask_range:
                if num > len(p):
                    continue
                masks.append(sentence_mask(cls_mask[batch_idx].unsqueeze(0), p[:num]))
            all_masks = torch.cat(masks)
            num_cand = len(all_masks)

            input_ids_rep = input_ids.repeat(num_cand, 1)
            kwargs = {
                'input_ids': input_ids_rep,
                'attention_mask': all_masks,
                'num_return_sequences': 1,
                'num_beams': 1,
                'length_penalty': 4.0,
                'max_length': 142,
                'min_length': 56,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
            }

            with torch.no_grad():
                pred_ids = self.model.generate(**kwargs)
            pred_str = [x.strip() for x in self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)]

            scores = []
            for idx, x in enumerate(pred_str):
                scores.append(self.compute_rouge([x], [reference], rouge_types=['rouge1'])['rouge1_f1'])
            labels = pred_ids
            labels[:, 0] = 0

            encoder_outputs = encoder_h[batch_idx].unsqueeze(0).repeat(num_cand, 1, 1).contiguous().clone()
            encoder_attention_mask = batch['attention_mask'][batch_idx].unsqueeze(0).repeat(num_cand, 1).contiguous()
            inputs = {
                'encoder_outputs': [encoder_outputs],
                'attention_mask': encoder_attention_mask,
                'labels': labels,
            }

            cand_outputs = self.model(**inputs, use_cache=False)
            nll = []
            for cand_idx in range(len(labels)):
                label_row = labels[cand_idx]
                loss_row_full = {'logits': cand_outputs['logits'][cand_idx]}
                nll.append(self.label_smoother(loss_row_full, label_row))

            ranks = np.argsort(-np.array(scores))

            # Compute Rank Score (not for back-prop)
            for a in range(num_cand - 1):
                pos_idx = ranks[a]
                for b in range(a + 1, num_cand):
                    neg_idx = ranks[b]
                    assert scores[neg_idx] <= scores[pos_idx]
                    pos_ll = - nll[pos_idx]
                    neg_ll = - nll[neg_idx]
                    if scores[neg_idx] == scores[pos_idx]:  # Don't train on equal / identical candidates
                        continue
                    margin = self.hparams.contrast_margin * (b - a)
                    margin_losses.append(torch.clamp(neg_ll - pos_ll + margin, min=0))
                    pos_neg_gap.append(float((pos_ll - neg_ll).detach().item()))

        avg_margin = torch.stack(margin_losses).mean()
        return avg_margin

    def training_step(self, batch, batch_idx):
        # Train on extractive labels
        plan_labels = batch.pop('plan_labels', None)
        priority = batch.pop('priority', None)
        plan_q = batch.pop('plan_q', None)
        cls_mask = batch.pop('cls_mask')
        reference = batch.pop('reference', None)

        if self.hparams.summary_style != 'score':
            output = self.model(**batch, use_cache=False)
            # Regular MLE decoder loss
            loss = output.loss
            self.log('train_loss', loss, on_epoch=False, on_step=True, prog_bar=True)
            # Return label-smoothed loss
            full_loss = self.label_smoother(output, batch['labels'])
            encoder_h = output.encoder_last_hidden_state
            cross_attentions = output.cross_attentions
            cross_att_head_pooled = [x.mean(dim=1) for x in cross_attentions]
            cross_att_pool_layers = torch.stack([x.max(dim=1)[0] for x in cross_att_head_pooled]).mean(dim=0)
        else:
            full_loss = 0
            output = self.model.model.encoder(**batch)
            encoder_h = output.last_hidden_state
            cross_att_pool_layers = None

        if plan_labels is not None:
            # Predict if a sentence is in oracle summary
            sent_preds = self.score_sents(cls_mask, encoder_h)

            if cross_att_pool_layers is not None:
                # cross_att_pool_layers
                sent_label_mask = (plan_labels == -100)
                cross_att_pool_layers.masked_fill_(sent_label_mask, float('-inf'))
                sent_preds.masked_fill_(sent_label_mask, float('-inf'))
                target_dist = torch.softmax(sent_preds, dim=1)

                batch_size = len(target_dist)
                klds = []
                for batch_idx in range(batch_size):
                    p = cross_att_pool_layers[batch_idx, cls_mask[batch_idx]]
                    q = target_dist[batch_idx, cls_mask[batch_idx]]
                    kld_loss = self.kld(p.unsqueeze(0) + 1e-4, q.unsqueeze(0))
                    klds.append(kld_loss)
                consistency_loss = torch.stack(klds).mean()
                self.log('train_consistency', consistency_loss, on_epoch=False, on_step=True, prog_bar=True)
                if self.hparams.add_consistency:
                    full_loss += consistency_loss

            if self.hparams.use_kld:
                extract_loss = self.compute_kld_extract_loss(sent_preds, plan_q, cls_mask)
            else:
                extract_loss = self.compute_ll_extract_loss(sent_preds, plan_labels)

            corel = self.compute_correlation(sent_preds, plan_q, cls_mask)
            self.log('train_corel', corel, on_epoch=False, on_step=True, prog_bar=True)
            self.log('train_extract', extract_loss, on_epoch=False, on_step=True, prog_bar=True)
            if full_loss is None:
                full_loss = extract_loss
            else:
                full_loss += extract_loss
                self.log('train_combined', full_loss, on_epoch=False, on_step=True, prog_bar=True)

            if self.hparams.add_brio:
                avg_margin = self.brio_step(self, priority, cls_mask, batch, reference, encoder_h)
                self.log('train_contrast', avg_margin, on_epoch=False, on_step=True, prog_bar=True)
                full_loss += avg_margin

        return full_loss

    def score_extracts(self, source, cls_mask, encoder_h, plan_labels, plan_q=None):
        batch_size = len(source)
        sent_preds = self.score_sents(cls_mask, encoder_h)
        extract_loss = self.compute_ll_extract_loss(sent_preds, plan_labels)
        # Only predict positively for CLS tokens
        sent_preds.masked_fill_(~ cls_mask, float('-inf'))
        y_hat_np = sent_preds.detach().cpu().numpy()
        extractive_summaries = [
            self.build_summaries(source=source[batch_idx], y_hat=y_hat_np[batch_idx, :])
            for batch_idx in range(batch_size)
        ]
        corel = None
        if plan_q is not None:
            corel = self.compute_correlation(sent_preds, plan_q, cls_mask)
        return extractive_summaries, extract_loss, corel

    def validation_step(self, batch, batch_idx):
        # Train on extractive labels
        plan_labels = batch.pop('plan_labels', None)
        plan_q = batch.pop('plan_q', None)
        cls_mask = batch.pop('cls_mask')
        batch_size = len(batch['input_ids'])
        references = batch['reference']

        source = self.parse_source_text_from_inputs(batch)

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
            encoder_kwargs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            output = self.model.model.encoder(**encoder_kwargs)
            encoder_h = output.last_hidden_state

        score_outputs = [None for _ in range(batch_size)]
        if 'score' in self.hparams.summary_style:
            extractive_summaries, extract_loss, corel = self.score_extracts(
                source, cls_mask, encoder_h, plan_labels, plan_q
            )
            for batch_idx in range(batch_size):
                score_outputs[batch_idx] = {
                    'source': source[batch_idx],
                    'abstracts': None, 'implied_extracts': None,
                    'extracts': [extractive_summaries[batch_idx]],
                    'reference': references[batch_idx],
                }
            self.log('val_corel', corel, on_epoch=True, on_step=False, prog_bar=True)
            self.log('val_extract', extract_loss, on_epoch=True, on_step=False, prog_bar=True)
            if loss is None:  # No generation
                loss = extract_loss
            else:
                loss += extract_loss
                self.log('val_combined', loss, on_epoch=True, on_step=False, prog_bar=True)

        gen_outputs = [None for _ in range(batch_size)]
        if self.hparams.summary_style != 'score':
            gen_outputs = self.shared_generate(batch, source, **validation_kwargs)

        # Merge the two if needed (score_abstract only)
        gen_outputs_resolved = self.merge_outputs(gen_outputs, score_outputs)

        all_metrics = {'val_loss': loss}
        # It's a list of dictionaries --> convert into dictionary of lists and process as a batch (for ROUGE)
        output_dict = defaultdict(list)
        for item in gen_outputs_resolved:
            for k, v in item.items():
                if type(v) == list:
                    output_dict[k] += v
                elif v is not None:
                    output_dict[k].append(v)
        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        if self.hparams.summary_style in {'abstract_plan', 'plan_abstract'}:
            from_oracle_rouge = self.generate_from_oracle(batch, reduce=True, **validation_kwargs)
            all_metrics.update(from_oracle_rouge)

        if len(output_dict['abstracts']) > 0:
            all_metrics.update(self.compute_rouge(output_dict['abstracts'], references))
            implied_extracts = [x['summary'] for x in output_dict['implied_extracts']]
            all_metrics.update(self.compute_rouge(implied_extracts, references, prefix='implied_'))

        if len(output_dict['extracts']) > 0:
            extracts = [x['summary'] for x in output_dict['extracts']]
            all_metrics.update(self.compute_rouge(extracts, references, prefix='extract_'))

        # Measure consistency between abstract (and implied extract) and generated extract
        if len(output_dict['abstracts']) > 0 and len(output_dict['extracts']) > 0:
            if 'score' not in self.hparams.summary_style:  # We aren't adhering to anything (they are separate)
                all_metrics.update(self.measure_plan_abstract_consistency(batch, output_dict, **validation_kwargs))
            # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
            # If the plan is working, this should be very high (the abstract should follow the plan)
            extracts = [x['summary'] for x in output_dict['extracts']]
            all_metrics.update(
                self.compute_rouge(extracts, output_dict['abstracts'], prefix='extract_gen_')
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

    def merge_outputs(self, gen_outputs, score_outputs):
        if self.hparams.summary_style == 'score':
            return score_outputs
        merged = []
        assert len(score_outputs) == len(gen_outputs)
        for batch_idx in range(len(score_outputs)):
            row = {}
            if gen_outputs[batch_idx] is not None:
                row.update(gen_outputs[batch_idx])
            if score_outputs[batch_idx] is not None:
                for k, v in score_outputs[batch_idx].items():
                    if v is None or row[k] is not None:
                        continue
                    row[k] = v
            merged.append(row)
        return merged

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        source = self.parse_source_text_from_inputs(batch)
        references = batch.pop('reference')
        cls_mask = batch.pop('cls_mask')
        plan_labels = batch.pop('plan_labels', None)
        plan_q = batch.pop('plan_q', None)
        use_hf_rouge = gen_kwargs.pop('use_hf_rouge')
        eval = not use_hf_rouge

        batch_size = len(references)
        gen_kwargs.update({
            'references': references
        })

        score_outputs = [None for _ in range(batch_size)]
        if 'score' in self.hparams.summary_style:
            assert 'score' in self.hparams.summary_style
            # Predict if a sentence is in oracle summary
            encoder_kwargs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            output = self.model.model.encoder(**encoder_kwargs)
            encoder_h = output.last_hidden_state
            extractive_summaries, extract_loss, corel = self.score_extracts(
                source, cls_mask, encoder_h, plan_labels, plan_q
            )
            for batch_idx in range(batch_size):
                score_outputs[batch_idx] = {
                    'source': source[batch_idx],
                    'abstracts': None, 'implied_extracts': None,
                    'extracts': [extractive_summaries[batch_idx]],
                    'reference': references[batch_idx],
                }

        gen_outputs = [None for _ in range(batch_size)]
        if self.hparams.summary_style != 'score':
            gen_outputs = self.shared_generate(batch, source, **gen_kwargs)

        # Merge the two if needed (score_abstract only)
        gen_outputs_resolved = self.merge_outputs(gen_outputs, score_outputs)

        batch_outputs = []
        from_oracle_metrics = None
        if self.hparams.summary_style in {'abstract_plan', 'plan_abstract'}:
            from_oracle_metrics = self.generate_from_oracle(batch, use_hf_rouge, **gen_kwargs)

        for batch_idx, (reference, gen_output) in enumerate(zip(references, gen_outputs_resolved)):
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
                'reference': reference, 'source': gen_output['source']['text'],
                'extract_idx': extract_idx_flat, 'implied_extract_idx': implied_extract_idx_flat,
            }

            if len(gen_output['extracts']) > 0 and 'sent_dist' in gen_output['extracts'][0]:
                dist_flat = '<cand>'.join([','.join(list(map(str, x['sent_dist']))) for x in gen_output['extracts']])
                save_out['sent_scores'] = dist_flat

            if from_oracle_metrics is not None:
                save_out.update(from_oracle_metrics[batch_idx])

            # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
            if gen_output['abstracts'] is not None:  # Take top of the beam or first returned sequence
                save_out.update(self.compute_rouge(gen_output['abstracts'][:1], [reference], eval=eval))
                implied_extracts = [x['summary'] for x in gen_output['implied_extracts']]
                save_out.update(self.compute_rouge(
                    implied_extracts[:1], [reference], prefix='implied_', eval=eval
                ))

                if len(gen_output['abstracts']) > 1:
                    # Get all the ROUGE abstracts (average ROUGE-1, ROUGE-2)
                    abstract_cand_metrics, best_abstract_metric = self.score_candidates(
                        [reference], gen_output['abstracts'], 'abstract', eval=eval
                    )
                    save_out.update(best_abstract_metric)
                    save_out['abstract_rouges'] = ','.join(
                        [str(x['best_abstract_mean_f1']) for x in abstract_cand_metrics]
                    )

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
                if len(extracts) > 1:
                    extract_cand_metrics, best_extract_metric = self.score_candidates(
                        [reference], extracts, 'extract', eval=eval
                    )
                    save_out.update(best_extract_metric)
                    save_out['extract_rouges'] = ','.join(
                        [str(x['best_extract_mean_f1']) for x in extract_cand_metrics]
                    )

            # Measure consistency between abstract (and implied extract) and generated extract
            if gen_output['abstracts'] is not None and gen_output['extracts'] is not None:
                if 'score' not in self.hparams.summary_style:  # We aren't adhering to anything (they are separate)
                    save_out.update(self.measure_plan_abstract_consistency(batch, gen_output, **gen_kwargs))
                # What is the ROUGE score of the extractive plan treating the abstractive prediction as the reference
                # If the plan is working, this should be very high (the abstract should follow the plan)
                extracts = [x['summary'] for x in gen_output['extracts']]
                save_out.update(
                    self.compute_rouge(extracts, gen_output['abstracts'], prefix='extract_gen_', eval=eval)
                )
            batch_outputs.append(save_out)

        return batch_outputs

    def shared_generate(self, batch, source, references, **gen_kwargs):
        default_kwargs = {  # Some of these values may get overridden by gen_kwargs
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_return_sequences': 1,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'output_scores': True
        }

        default_kwargs.update(gen_kwargs)
        pred_ids = self.model.generate(**default_kwargs)
        gold_ids = batch['labels']
        gold_ids[torch.where(batch['labels'] == -100)] = 1
        input_ids = batch['input_ids']

        batch_size = len(input_ids)
        num_pred = len(pred_ids)
        num_cands = gen_kwargs['num_return_sequences']
        assert num_cands * batch_size == num_pred
        pred_ids = pred_ids.view(batch_size, num_cands, -1) if num_cands > 1 else pred_ids.unsqueeze(1)
        return [
            self.parse_output(source[batch_idx], references[batch_idx], pred_ids[batch_idx])
            for batch_idx in range(batch_size)
        ]

    def ensure_extract(self, pred_str, source):
        extractive = []
        idxs = []
        pred_sents = convert_to_sents(pred_str, self.nlp)
        for sent in pred_sents:
            sent_toks = [str(token.text).strip() for token in sent if len(str(token.text).strip()) > 0]
            if len(sent_toks) <= 1:
                continue
            max_intersect = 0
            closest_sent = ''
            best_idx = -1
            for source_idx in range(len(source['sents'])):
                num_intersect = len(set(source['sent_toks'][source_idx]).intersection(set(sent_toks)))
                if num_intersect >= max_intersect:
                    closest_sent = source['sents'][source_idx]
                    max_intersect = num_intersect
                    best_idx = source_idx
            idxs.append(best_idx)
            extractive.append(str(closest_sent))
        return {'summary': ' '.join(extractive), 'idxs': idxs}

    def get_summary_from_sent_idxs(self, source, extractive_idxs, sort=False):
        if sort:
            extractive_idxs = list(sorted(extractive_idxs))
        summary = ' '.join([source['sents'][i].strip() for i in extractive_idxs])
        return {
            'idxs': extractive_idxs,
            'summary': summary
        }

    def parse_output(self, source, reference, pred_ids):
        pred_str = self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        num_pred = len(pred_str)
        if 'plan' in self.hparams.summary_style:
            decoded_sent_preds = [
                re.findall(r'(<s\d+>)', x) for x in
                self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=False)
            ]

            decoded_sent_idxs = [
                [int(re.search(r'\d+', tag)) for tag in sent_tags] for sent_tags in decoded_sent_preds
            ]

            extracts = [self.build_summaries(source, sent_idx) for sent_idx in decoded_sent_idxs]
            abstracts = None if self.hparams.summary_style == 'plan' else pred_str
        elif self.hparams.summary_style == 'extract':  # Our abstractive generation is actually an extract
            extracts = list(map(lambda i: self.ensure_extract(pred_str[i], source), range(num_pred)))
            abstracts = None
        elif self.hparams.summary_style == 'score_abstract':
            abstracts = pred_str
            extracts = None  # these will be filled in after (not part of generation)
        elif self.hparams.summary_style == 'abstract':
            extracts = None
            abstracts = pred_str
        elif self.hparams.summary_style == 'hybrid_control':
            # TODO signal if extractive or abstractive
            is_extractive = ['<extract>' in p for p in ['<extract>']]
            extracts = [self.ensure_extract(pred_str[i], source) for i in range(len(is_extractive)) if is_extractive[i]]
            abstracts = [pred_str[i] for i in range(len(is_extractive)) if not is_extractive[i]]
            if len(extracts) == 0:
                extracts = None
            if len(abstracts) == 0:
                abstracts = None
        else:
            raise Exception(f'Unrecognized summary style -> {self.hparams.summary_style}')

        implied_extracts = None
        if abstracts is not None:
            implied_extracts = []
            abstract_sents = [convert_to_sents(x, self.nlp) for x in abstracts]
            abstract_sents_tok = [[[
                str(token.text) for token in sentence] for sentence in abstract_sent] for abstract_sent in
                abstract_sents]

            for idx in range(num_pred):
                implied_oracle_idx = gain_selection(
                    source['sent_toks'], abstract_sents_tok[idx], 5, lower=True, sort=True
                )[0]
                implied_oracle = ' '.join([str(source['sents'][i]) for i in implied_oracle_idx])
                implied_extracts.append({
                    'idxs': implied_oracle_idx,
                    'summary': implied_oracle,
                })

        return {
            'source': source,
            'reference': reference,
            'abstracts': abstracts,
            'extracts': extracts,
            'implied_extracts': implied_extracts,
        }

    def measure_plan_abstract_consistency(self, batch, outputs, eval=False, **gen_kwargs):
        extract_idxs = [x['idxs'] for x in outputs['extracts']]
        implied_extract_idxs = [x['idxs'] for x in outputs['implied_extracts']]

        extracts = [x['summary'] for x in outputs['extracts']]
        implied_extracts = [x['summary'] for x in outputs['implied_extracts']]

        metrics = self.compute_rouge(extracts, implied_extracts, eval=eval, prefix='extract_implied_')

        overlaps = []
        for batch_idx, (extract_idx, implied_idx) in enumerate(zip(extract_idxs, implied_extract_idxs)):
            n = len(outputs['source'][batch_idx]['sents'])
            intersection = set(implied_idx).intersection(extract_idx)
            overlap_p = len(intersection) / len(extract_idx)
            overlap_r = len(intersection) / len(implied_idx)
            overlap_f1 = 0.0 if max(overlap_p, overlap_r) == 0 else 2 * overlap_p * overlap_r / (
                    overlap_p + overlap_r)

            row = {
                'extract_implied_sent_precision': overlap_p,
                'extract_implied_sent_recall': overlap_r,
                'extract_implied_sent_f1': overlap_f1
            }

            if 'score' not in self.hparams.summary_style:  # We aren't adhering to anything (they are separate)
                rand_plan_idxs = list(np.sort(list(np.random.choice(np.arange(n), size=(min(n, 3)), replace=False))))
                rand_plan = ''.join([f'<s{i}>' for i in rand_plan_idxs])
                decoder_input_ids = self.tokenizer(
                    '<s></s>' + rand_plan + '<sep>', add_special_tokens=False, return_tensors='pt'
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

                pred_sents = convert_to_sents(pred, self.nlp)
                from_rand_abstract_sents_tok = [[str(token.text) for token in sentence] for sentence in pred_sents]

                source_toks = outputs['source'][batch_idx]['sent_toks']
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

        return avgs

    def compute_rouge(self, generated, gold, prefix='', eval=False, rouge_types=['rouge1', 'rouge2', 'rougeL']):
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
