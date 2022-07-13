import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import AutoTokenizer, BartTokenizer

from gen_transformers.dataset import SummaryDataModule
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus, set_same_seed

from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from transformers.utils import ModelOutput

class EncoderOutputs(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    attentions: Optional[torch.FloatTensor] = None
    hidden_states: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

parser = argparse.ArgumentParser('BART/PEGASUS trainer.')

# Configuration Parameters
parser.add_argument('-debug', default=False, action='store_true')
parser.add_argument('--experiment', default='default')
parser.add_argument('--dataset', default='cnn_dailymail')
parser.add_argument('--restore_path', default=None)
parser.add_argument('--seed', default=1992, type=int)
parser.add_argument('--max_gpus', default=1, type=int)
parser.add_argument('-cpu', default=True, action='store_true')
parser.add_argument('--max_val_examples', default=1024, type=int)
parser.add_argument('--gpu_device', default=None, type=int)
parser.add_argument('--data_dir', default='./data')
parser.add_argument('-no_schedule', default=False, action='store_true')
parser.add_argument('-offline', default=False, action='store_true')
parser.add_argument('-find_lr', default=False, action='store_true')
# How many processes to use when loading batches on CPU
parser.add_argument('--num_dataloaders', default=8, type=int)
parser.add_argument('-extract_indicators', default=False, action='store_true')
# How many sentences to make visible to the decoder (5 is randomly set based on summary lengths of ~2-5 sentences)
parser.add_argument('--copy_bart_class_dropout', default=0.0, type=float)
parser.add_argument('-add_sent_brio', default=False, action='store_true')
parser.add_argument('--contrast_margin', default=0.01, type=float)
parser.add_argument('--brio_loss_coef', default=1, type=float)

# Hyper-Parameters
parser.add_argument('--lr', type=float, default=1e-5)  # used to be 2.2e-4
parser.add_argument('--weight_decay', type=float, default=5e-5)
# Gradient accumulation will adjust for the ratio between target_batch_size and per_device_train_bs
parser.add_argument('--target_batch_size', type=int, default=16)
parser.add_argument('--per_device_train_bs', type=int, default=8)
parser.add_argument('--per_device_eval_bs', type=int, default=16)
parser.add_argument('--warmup_steps', type=int, default=200)
parser.add_argument('--max_steps', default=150000, type=int)
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--max_output_length', type=int, default=256)  # For training only
parser.add_argument('--max_num_sents', type=int, default=200)
parser.add_argument('--max_input_length', type=int, default=1024)
parser.add_argument('--train_frac', type=float, default=0.1)
parser.add_argument('-add_align', action='store_true', default=False)
parser.add_argument('--extract_method', type=str, default='select', choices=['generate', 'select'])
parser.add_argument('--pretrained_path', default=None, help='Path to a pre-trained TransformerSummarizer model.')
# HuggingFace identifier of model for which to load weights for fine-tuning
parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
    'facebook/bart-base',
    'facebook/bart-large',
    'Yale-LILY/brio-cnndm-uncased',
])

# Task-specific / Project-specific parameters
parser.add_argument(
    '--summary_style',
    default='extract_abstract',
    choices=[
        'extract_abstract',
        'abstract',
        'extract'
    ], help='Target output during training. plan is a sequence of <s{idx}> tokens, extract is oracle summary, '
            'abstract is original reference'
)
# This will be automatically determine by summary_style (i.e., 'plan' or not)
parser.add_argument('-add_sent_toks', default=False, action='store_true')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

# Won't held yet for multi-gpu
args.grad_accum = args.target_batch_size // args.per_device_train_bs

if args.debug:  # Use small data and tiny BART model
    args.hf_model = 'sshleifer/bart-tiny-random'

# Override: If we are generating a sentence plan, we MUST include <s{idx}> tokens in the source input
args.add_sent_toks = args.add_sent_toks or 'extract' in args.summary_style or args.extract_indicators
if args.add_sent_toks:
    print('Pre-pending each sentence in the source document with special token <s{idx}>.')

args.weight_dir = os.path.join(args.data_dir, 'weights')
print(f'Setting up {args.weight_dir} to store model weights, metrics, and results.')
os.makedirs(args.weight_dir, exist_ok=True)

# Set same random seed for each run
set_same_seed(args.seed)

if args.gpu_device is not None:
    gpus = [args.gpu_device]
else:
    gpus = get_free_gpus() if torch.cuda.is_available() and not args.cpu else None
    assert gpus is None or len(gpus) > 0
    if gpus is not None and (args.debug or args.find_lr):
        gpus = [gpus[0]]
    if gpus is not None and len(gpus) > args.max_gpus:
        gpus = gpus[:args.max_gpus]
    if gpus is not None:
        gpu_str = ','.join([str(x) for x in gpus])
        print(f'Using GPUS --> {gpu_str}...')

args.num_gpus = None if gpus is None else len(gpus)
print('Num GPUs --> {}'.format(args.num_gpus))
precision = 16 if args.num_gpus is not None else 32
experiment_dir = os.path.join(args.weight_dir, args.experiment)
os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable
if 'brio' in args.hf_model:
    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)

if args.add_sent_toks:
    add_tokens = [f'<s{i}>' for i in range(args.max_num_sents)]
    add_tokens.append('<sep>')  # Not used right now
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

tokenizer_dir = os.path.join(experiment_dir, 'tokenizer')
if not args.debug:
    tokenizer.save_pretrained(tokenizer_dir)
if args.pretrained_path is None:
    model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model)
    val_check_interval = 1.0 if args.debug or args.train_frac <= 0.2 else 0.25
else:
    tok_path = '/'.join(args.pretrained_path.split('/')[:-4]) + '/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=args.pretrained_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
    )

    if args.add_sent_toks and '<s1>' not in tokenizer.additional_special_tokens:
        add_tokens = [f'<s{i}>' for i in range(args.max_num_sents)]
        add_tokens.append('<sep>')  # Not used right now
        special_tokens_dict = {'additional_special_tokens': add_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.model.resize_token_embeddings(len(tokenizer))

    model.hparams.add_sent_brio = False  # args.add_sent_brio
    model.hparams.extract_indicators = args.extract_indicators
    model.hparams.contrast_margin = args.contrast_margin
    model.hparams.brio_loss_coef = args.brio_loss_coef
    val_check_interval = 0.25  # Depending on what you're doing you should change this
datamodule = SummaryDataModule(args, tokenizer=tokenizer)

logger = pl_loggers.WandbLogger(
    name=args.experiment,
    save_dir=experiment_dir,
    offline=args.debug or args.offline,
    project='faith_sum',
    entity='mertketenci',
)

monitor_metric = 'val_combined'
mode = 'min'
checkpoint_callback = ModelCheckpoint(
    monitor=monitor_metric,
    save_top_k=1,
    save_last=False,
    mode=mode
)
# early_stopping = EarlyStopping(monitor_metric, patience=20, verbose=True)
callbacks = [checkpoint_callback]
if not (args.no_schedule or args.debug or args.find_lr):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
plugins = DDPPlugin(find_unused_parameters=False) if args.num_gpus is not None and args.num_gpus > 1 else None
trainer = pl.Trainer.from_argparse_args(
    args,
    resume_from_checkpoint=args.restore_path,
    callbacks=callbacks,
    logger=logger,
    precision=precision,
    accelerator=None if args.num_gpus is None or args.num_gpus == 1 else 'ddp',
    gpus=gpus,
    default_root_dir=experiment_dir,
    gradient_clip_val=0.1,
    accumulate_grad_batches=args.grad_accum,
    val_check_interval=val_check_interval,
    check_val_every_n_epoch=args.max_epochs if args.debug else 1,
    num_sanity_val_steps=0 if args.debug else 2,
    log_every_n_steps=25,
    max_steps=args.max_steps,
    plugins=plugins,
    # detect_anomaly=args.debug
)

input_ids: torch.LongTensor = None
# cls_mask: torch.BoolTensor = None
extract_indicators: Optional[torch.Tensor] = None
attention_mask: Optional[torch.Tensor] = None
decoder_input_ids: Optional[torch.LongTensor] = None
decoder_attention_mask: Optional[torch.LongTensor] = None
head_mask: Optional[torch.Tensor] = None
decoder_head_mask: Optional[torch.Tensor] = None
cross_attn_head_mask: Optional[torch.Tensor] = None
encoder_outputs: Optional[List[torch.FloatTensor]] = None
past_key_values: Optional[List[torch.FloatTensor]] = None
inputs_embeds: Optional[torch.FloatTensor] = None
decoder_inputs_embeds: Optional[torch.FloatTensor] = None
use_cache: Optional[bool] = None
output_attentions: Optional[bool] = None
output_hidden_states: Optional[bool] = None
return_dict: Optional[bool] = None

train_dataloader = datamodule.train_dataloader(2)

batch = next(iter(train_dataloader))

batch.pop('references')
batch.pop('oracle_labels')

input_ids = batch['input_ids']
attention_mask = batch['attention_mask']
cls_mask = batch['cls_mask']
labels = batch['labels']
model = model.model
# outputs = model(**batch)
# self = model
# self = self.model
# model.set_cls_mask(cls_mask)
# num_beams = 2
# print(tokenizer.batch_decode(model.generate(input_ids, num_beams=num_beams)))

optimizer = torch.optim.Adam(
    list(model.parameters()),
    lr=1e-3
)

optimizer.zero_grad()
loss = model(**batch).loss
loss.backward()
optimizer.step()
