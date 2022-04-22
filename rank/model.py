import itertools
import os

import pytorch_lightning as pl
import numpy as np
from scipy.stats import pearsonr
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


class SummaryRanker(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr  # Necessary for tune_lr to work with PytorchLightning
        self.label_smoother = LabelSmoother(epsilon=0.1)
        # <sep> separates plan from abstract for plan models
        self.special_id_cutoff = self.tokenizer.convert_tokens_to_ids('<sep>')

    def compute_ll(self, batch, labels, encoder_h):
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
        idx_combos = list(itertools.combinations(range(num_cand), 2))
        mrr = []
        score_adv = []
        corels = []
        pos_neg_gap = []
        for batch_idx in range(batch_size):
            scores = batch_scores[batch_idx]
            nll = batch_nll[batch_idx * num_cand: (batch_idx + 1) * num_cand]
            # For computing PearsonR
            ll = [-float(x.detach().item()) for x in nll]

            max_score = max(scores)
            best_idxs = [tmp_idx for tmp_idx in range(len(scores)) if scores[tmp_idx] == max_score]
            pred_ranks = np.argsort(nll)
            corels.append(float(pearsonr(ll, scores)[0]))

            best_rank = -1
            for rank, pred_rank in enumerate(pred_ranks):
                if pred_rank in best_idxs:
                    best_rank = rank + 1
                    break
            mrr.append(1 / best_rank)
            score_adv.append(scores[np.argmax(ll)] - np.mean(scores))

            # Compute Rank Score (not for back-prop)
            for a_idx, b_idx in idx_combos:
                if scores[a_idx] > scores[b_idx]:
                    pos_idx = a_idx
                    neg_idx = b_idx
                else:
                    pos_idx = b_idx
                    neg_idx = a_idx
                assert scores[neg_idx] <= scores[pos_idx]
                pos_ll = - nll[pos_idx]
                neg_ll = - nll[neg_idx]
                if scores[neg_idx] == scores[pos_idx]:  # Don't train on equal / identical candidates
                    continue
                margin_losses.append(torch.clamp(neg_ll - pos_ll + self.hparams.contrast_margin, min=0))
                pos_neg_gap.append(float((pos_ll - neg_ll).detach().item()))
        return margin_losses, pos_neg_gap, corels, score_adv, mrr

    def shared_step(self, batch, split='train'):
        is_train = split == 'train'
        batch_size = len(batch['input_ids'])
        num_cand = len(batch['labels']) // batch_size

        # Perturbed Plans
        scores = batch.pop('scores', None)

        encoder_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        encoder_h = self.model.model.encoder(**encoder_inputs).last_hidden_state

        # Single positive is ground-truth plan and abstractive reference
        _, _, batch_nll = self.compute_ll(batch, batch['labels'], encoder_h)

        # Extractive Plan Contrast Loss
        # Train log likelihood of generating ground-truth plan to be higher than perturbed
        batch_margins, pos_neg_gap, corel, score_adv, mrr = self.contrast_loss(batch_size, num_cand, batch_nll, scores)

        # Average both and add to MLE language model loss depending on if plan and/or abstract in --contrast_modes
        avg_margin = torch.stack(batch_margins).mean()
        avg_score_adv = np.stack(score_adv).mean()
        avg_mrr = np.stack(mrr).mean()
        avg_corel = np.stack(corel).mean()
        avg_pos_neg_gap = np.stack(pos_neg_gap).mean()
        self.log(f'{split}_margin', avg_margin, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_gap', avg_pos_neg_gap, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_corel', avg_corel, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_advantage', avg_score_adv, on_epoch=not is_train, on_step=is_train, prog_bar=True)
        self.log(f'{split}_mrr', avg_mrr, on_epoch=not is_train, on_step=is_train, prog_bar=True)
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
