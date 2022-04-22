from collections import defaultdict
import itertools
import os

import pytorch_lightning as pl
import numpy as np
from scipy.stats import pearsonr
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


class SummaryRanker(pl.LightningModule):
    def __init__(self, args, tokenizer, finetuned_model):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = finetuned_model
        self.lr = self.hparams.lr  # Necessary for tune_lr to work with PytorchLightning
        self.label_smoother = LabelSmoother(epsilon=0.1)
        # <sep> separates plan from abstract for plan models
        self.special_id_cutoff = self.tokenizer.convert_tokens_to_ids('<sep>')

    def compute_ll(self, batch, encoder_h):
        labels = batch['labels']
        batch_size = len(encoder_h)
        num_cand = len(labels) // batch_size
        # Ignore the <eos> token
        # Tile this for each num_neg or num_pos decoder target
        model_dim = self.model.config.d_model  # 768 for BART (24 for debug shleifer/tiny-random)
        encoder_outputs = encoder_h.unsqueeze(1).repeat(
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
        full_nll, plan_nll, abstract_nll = [], [], []
        for cand_idx in range(len(labels)):
            label_row = labels[cand_idx]
            cutoff = int(torch.where(label_row == self.special_id_cutoff)[0])

            label_row_plan = label_row[:cutoff]
            loss_row_plan = {'logits': outputs['logits'][cand_idx, : cutoff, :]}
            plan_nll.append(self.label_smoother(loss_row_plan, label_row_plan))

            label_row_abstract = label_row[cutoff + 1:]
            loss_row_abstract = {'logits': outputs['logits'][cand_idx, cutoff + 1:, :]}
            abstract_nll.append(self.label_smoother(loss_row_abstract, label_row_abstract))

            loss_row_full = {'logits': outputs['logits'][cand_idx]}
            full_nll.append(self.label_smoother(loss_row_full, label_row))

        return plan_nll, abstract_nll, full_nll

    def contrast_loss(self, batch_size, num_cand, batch_nll, batch_scores):
        """
        :param batch_size: Batch Size
        :param num_cand: Number of targets for each example in batch (batch_size * num_neg_cand total)
        :param batch_nll_pos: Negative Log Likelihoods for generating the 'positive' ground-truth
        :param batch_nll_neg: Negative Log Likelihoods for generating the 'negative' ground-truth
        :return: Margin losses for each pair of pos and neg.  Positive targets should have a higher LL than negatives,
        with a margin defined by --contrast_margin.
        """
        margin_losses = []
        mrr = []
        score_adv = []
        pred_scores = []
        corels = defaultdict(list)
        pos_neg_gap = []
        for batch_idx in range(batch_size):
            avg_scores = batch_scores['avg'][batch_idx]
            extract_scores = batch_scores['extract'][batch_idx]
            abstract_scores = batch_scores['abstract'][batch_idx]
            full_nll = batch_nll['full'][batch_idx * num_cand: (batch_idx + 1) * num_cand]
            plan_nll = batch_nll['plan'][batch_idx * num_cand: (batch_idx + 1) * num_cand]
            abstract_nll = batch_nll['abstract'][batch_idx * num_cand: (batch_idx + 1) * num_cand]
            # For computing PearsonR
            full_ll = [-float(x.detach().item()) for x in full_nll]
            plan_ll = [-float(x.detach().item()) for x in plan_nll]
            abstract_ll = [-float(x.detach().item()) for x in abstract_nll]

            corel_recipes = [
                ('full_ll_avg', full_ll, avg_scores),
                ('plan_ll_avg', plan_ll, avg_scores),
                ('abstract_ll_avg', abstract_ll, avg_scores),
                ('full_ll_extract', full_ll, extract_scores),
                ('plan_ll_extract', plan_ll, extract_scores),
                ('abstract_ll_extract', abstract_ll, extract_scores),
                ('full_ll_abstract', full_ll, abstract_scores),
                ('plan_ll_abstract', plan_ll, abstract_scores),
                ('abstract_ll_abstract', abstract_ll, abstract_scores),
            ]

            for key, a, b in corel_recipes:
                corels[key].append(float(pearsonr(a, b)[0]))

            # How does the model rank the best (by oracle ROUGE) candidate.
            pred_ranks = np.argsort(full_nll)
            best_rank = int(np.where(pred_ranks == 0)[0]) + 1  # Best is the first candidate due to sorting in dataset.
            mrr.append(1 / best_rank)

            # What is the ROUGE score of our prediction (highest LL)
            pred_score = avg_scores[np.argmax(full_ll)]
            pred_scores.append(pred_score)
            # How much better is this than the average score (adv = advantage)
            score_adv.append(avg_scores[np.argmax(full_ll)] - np.mean(avg_scores))

            # Compute Rank Score (not for back-prop)
            for pos_idx in range(num_cand):
                for neg_idx in range(pos_idx + 1, num_cand):
                    assert avg_scores[neg_idx] <= avg_scores[pos_idx]
                    pos_ll = - full_nll[pos_idx]
                    neg_ll = - full_nll[neg_idx]
                    if avg_scores[neg_idx] == avg_scores[pos_idx]:  # Don't train on equal / identical candidates
                        continue
                    margin = self.hparams.contrast_margin * (neg_idx - pos_idx)
                    margin_losses.append(torch.clamp(neg_ll - pos_ll + margin, min=0))
                    pos_neg_gap.append(float((pos_ll - neg_ll).detach().item()))
        return margin_losses, pos_neg_gap, corels, pred_scores, score_adv, mrr

    def shared_step(self, batch, split='train'):
        is_train = split == 'train'
        batch_size = len(batch['input_ids'])
        num_cand = len(batch['labels']) // batch_size

        # Perturbed Plans
        extract_scores = batch.pop('extract_scores', None)
        abstract_scores = batch.pop('abstract_scores', None)
        avg_scores = batch.pop('avg_scores', None)
        scores = {
            'extract': extract_scores,
            'abstract': abstract_scores,
            'avg': avg_scores
        }

        encoder_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        encoder_h = self.model.model.encoder(**encoder_inputs).last_hidden_state

        # Single positive is ground-truth plan and abstractive reference
        plan_nll, abstract_nll, full_nll = self.compute_ll(batch, encoder_h)
        batch_nll = {
            'plan': plan_nll,
            'abstract': abstract_nll,
            'full': full_nll  # negative log likelihood of generating the plan + abstract
        }

        # Extractive Plan Contrast Loss
        # Train log likelihood of generating ground-truth plan to be higher than perturbed
        batch_margins, pos_neg_gap, corels, pred_scores, score_adv, mrr = self.contrast_loss(
            batch_size, num_cand, batch_nll, scores
        )

        # Average both and add to MLE language model loss depending on if plan and/or abstract in --contrast_modes
        avg_margin = torch.stack(batch_margins).mean()
        avg_pred_score = np.stack(pred_scores).mean()
        avg_score_adv = np.stack(score_adv).mean()
        avg_mrr = np.stack(mrr).mean()
        avg_pos_neg_gap = np.stack(pos_neg_gap).mean()
        self.log(f'{split}_margin', avg_margin, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_gap', avg_pos_neg_gap, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_avg_rouge', avg_pred_score, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_advantage', avg_score_adv, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_mrr', avg_mrr, on_epoch=not is_train, on_step=is_train, prog_bar=True)

        for k, v in corels.items():
            avg_corel = np.stack(v).mean()
            self.log(f'{split}_{k}_corel', avg_corel, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        return avg_margin

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, split='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, split='val')

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
