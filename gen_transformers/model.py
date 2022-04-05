import regex as re

from datasets import load_metric
import pytorch_lightning as pl
import nltk
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother


def postprocess_text(texts):
    return ['\n'.join(nltk.sent_tokenize(text.strip())) for text in texts]


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        """
        bart_model -> can load in pre-trained bart weights outside of this function
        """
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
        ])
        return parent_parser

    def training_step(self, batch, batch_idx):
        output = self.model(**batch, use_cache=False)
        loss = output.loss
        self.log('train_loss', loss, on_epoch=False, on_step=True, prog_bar=True)
        smooth_loss = self.label_smoother(output, batch['labels'])
        return smooth_loss

    def rouge_metrics(self, generated, gold):
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        rouge_output = self.rouge.compute(predictions=generated, references=gold, rouge_types=rouge_types)

        stats = {}
        f1s = []
        for rouge_type in rouge_types:
            stats[f'{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
            stats[f'{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
            stats[f'{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
            f1s.append(rouge_output[rouge_type].mid.fmeasure)
        stats['mean_f1'] = np.array(f1s).mean()
        return stats

    def shared_generate(self, batch, **gen_kwargs):
        kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_return_sequences': 1,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }
        kwargs.update(gen_kwargs)
        generated_ids = self.model.generate(**kwargs)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        generated_str = list(map(lambda x: x.strip(':'), generated_str))
        output_ids = batch['labels']
        output_ids[torch.where(batch['labels'] == -100)] = 1
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        # TODO remove when this becomes special token
        gold_str = list(map(lambda x: x.replace('<sep>', '').strip(), gold_str))
        predicted_extracts = None
        if self.hparams.add_sent_toks:
            decoded_sent_preds = [
                re.findall(r'(<s\d+>)', x) for x in
                self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=False)
            ]
            decoded_inputs = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
            predicted_extracts = []
            for example_idx, decoded_input in enumerate(decoded_inputs):
                source_sent_tps = re.split(r'(<s\d+>)', decoded_input)
                first_sent = source_sent_tps[2]
                predicted_extractive_summary = []
                for sent_idx, tp in enumerate(source_sent_tps):
                    if tp in decoded_sent_preds[example_idx]:
                        predicted_extractive_summary.append(source_sent_tps[sent_idx + 1].strip())
                if len(predicted_extractive_summary) == 0:  # Default to LEAD-1 if none predicted
                    predicted_extracts.append(first_sent)
                else:
                    predicted_extracts.append(' '.join(predicted_extractive_summary))
            predicted_extracts = postprocess_text(predicted_extracts)
        generated_str = postprocess_text(generated_str)
        gold_str = postprocess_text(gold_str)
        return generated_str, gold_str, predicted_extracts

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        validation_kwargs = {
            'num_beams': 1,
        }

        generated_str, gold_str, extracted_str = self.shared_generate(batch, **validation_kwargs)
        metrics = self.rouge_metrics(generated_str, gold_str)
        for k, v in metrics.items():
            if v is None:
                continue
            self.log(k, v, on_epoch=True, on_step=False, prog_bar=True)

        if extracted_str is not None:
            extracted_metrics = self.rouge_metrics(extracted_str, gold_str)
            for k, v in extracted_metrics.items():
                if v is None:
                    continue
                self.log(f'extract_{k}', v, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

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

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        generated_str, gold_str, extracted_str = self.shared_generate(batch, **gen_kwargs)
        source = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=True)
        outputs = self.rouge_metrics(generated_str, gold_str)
        outputs['source'] = source
        outputs['prediction'] = generated_str
        outputs['target'] = gold_str
        if extracted_str is not None:
            outputs['extracted_str'] = extracted_str
            extracted_metrics = self.rouge_metrics(extracted_str, gold_str)
            for k, v in extracted_metrics.items():
                if v is None:
                    continue
                outputs[f'extract_{k}'] = v
        return outputs

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
