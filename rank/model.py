import os

import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import RobertaForSequenceClassification, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')


class SummaryRanker(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.config = AutoConfig.from_pretrained('roberta-base')
        self.config.num_labels = 1
        self.config.num_features = 5
        self.config.extract_indicators = True  # Embeddings for extract tokens
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = None if not hasattr(self.hparams, 'lr') else self.hparams.lr
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def shared_step(self, batch, split='train'):
        batch.pop('dataset_idx', None)
        is_train = split == 'train'
        scores = batch['scores']
        score_dist = batch['score_dist']
        batch_inputs = batch['inputs']
        batch_size = len(scores)
        num_candidates = len(batch_inputs['input_ids']) // batch_size
        outputs = self.model(**batch_inputs)
        logits = outputs.logits

        logits = logits.view(batch_size, num_candidates)
        pred_dist = torch.softmax(logits, dim=1)

        expected_rouge = (pred_dist * score_dist).sum(dim=1)
        expected_rouge = torch.log(expected_rouge + 1e-4).mean(dim=0)

        labels = torch.zeros(size=(batch_size, ), device=self.device).long()
        loss = self.loss(logits, labels)
        argmax_logits = logits.argmax(dim=1)
        pred_scores = np.array([scores[batch_idx][argmax_logits[batch_idx].item()] for batch_idx in range(batch_size)])
        pred_scores_mean = np.mean(pred_scores)

        corels = []
        for batch_idx in range(batch_size):
            corels.append(pearsonr(scores[batch_idx], pred_dist[batch_idx].detach().cpu().numpy())[0])
        avg_corel = np.mean(corels)

        self.log(f'{split}/loss', loss, on_step=is_train, on_epoch=not is_train, prog_bar=True)
        self.log(f'{split}/corel', avg_corel, on_step=is_train, on_epoch=not is_train, prog_bar=True)
        self.log(f'{split}/expected_rouge', expected_rouge, on_step=is_train, on_epoch=not is_train, prog_bar=True)
        self.log(f'{split}/score', pred_scores_mean, on_step=is_train, on_epoch=not is_train, prog_bar=True)
        return -expected_rouge

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, split='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, split='val')

    def predict_step(self, batch):
        batch_inputs = batch['inputs']
        n = len(batch_inputs['input_ids'])
        dataset_idx = batch.pop('dataset_idx')
        batch_size = len(dataset_idx)
        assert n % batch_size == 0
        num_candidates = n // batch_size
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            logits = outputs.logits
            logits = logits.view(batch_size, num_candidates)
            pred_dist = torch.softmax(logits, dim=1).cpu().numpy()
            return dataset_idx, pred_dist

    def configure_optimizers(self):
        nps = list(self.named_parameters())

        classifier = [(n, p) for n, p in nps if 'classifier' in n and p.requires_grad]
        embeds = [(n, p) for n, p in nps if 'extract_indicator_embeddings' in n and p.requires_grad]
        used_names = set([x for (x, _) in classifier + embeds])
        remaining = [(n, p) for n, p in nps if n not in used_names and p.requires_grad]
        grouped_parameters = [
            {
                'params': [p for n, p in remaining],
                'weight_decay': self.hparams.weight_decay,
                'lr': self.lr,
            },
            {
                'params': [p for n, p in classifier],
                'weight_decay': 0.0,
                'lr': 1e-3,
            },
            {
                'params': [p for n, p in embeds],
                'weight_decay': 0.0,
                'lr': 1e-3
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)
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
