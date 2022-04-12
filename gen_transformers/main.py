import os
from pathlib import Path
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


def get_path_from_exp(weights_dir, experiment):
    dir = os.path.join(weights_dir, experiment)
    paths = list(Path(dir).rglob('*.ckpt'))
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')


def run(args):
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
    num_machine = 1 if args.num_gpus is None else args.num_gpus
    denom = args.per_device_train_bs * num_machine
    assert args.target_batch_size % denom == 0
    args.grad_accum = int(denom / (args.per_device_train_bs * num_machine))
    print('Num GPUs --> {}'.format(args.num_gpus))
    precision = 16 if args.num_gpus is not None else 32
    experiment_dir = os.path.join(args.weight_dir, args.experiment)
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable
    if 'brio' in args.hf_model:
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)

    if args.add_sent_toks:
        add_tokens = [f'<s{i}>' for i in range(100)]
        add_tokens.append('<sep>')
        special_tokens_dict = {'additional_special_tokens': add_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)

    if args.fragments:
        special_tokens_dict = {'additional_special_tokens': ['<frag>', '<sep>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    if args.summary_style == 'hybrid_control':
        special_tokens_dict = {'additional_special_tokens': ['<extract>', '<abstract>']}
        tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer_dir = os.path.join(experiment_dir, 'tokenizer')
    if not args.debug:
        tokenizer.save_pretrained(tokenizer_dir)
    if args.pretrained_path is None:
        model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model)
    else:
        model = TransformerSummarizer.load_from_checkpoint(
            checkpoint_path=args.pretrained_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=True
        )
    datamodule = SummaryDataModule(args, tokenizer=tokenizer)

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='faith_sum',
        entity='griffinadams',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        save_last=False,
        mode='min'
    )
    early_stopping = EarlyStopping('val_loss', patience=5, verbose=True)
    callbacks = [checkpoint_callback, early_stopping]
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
        val_check_interval=1.0 if args.debug else 0.25,
        check_val_every_n_epoch=args.max_epochs if args.debug else 1,
        num_sanity_val_steps=2,
        log_every_n_steps=50,
        max_steps=args.max_steps,
        plugins=plugins
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-4, max_lr=1e-2, update_attr=True, num_training=100)
        print(lr_finder.results)
    else:
        print('Starting training...')
        trainer.fit(model, datamodule=datamodule)
        print(f'Best weights saved --> {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BART/PEGASUS trainer.')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--restore_path', default=None)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--max_gpus', default=1, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-no_schedule', default=False, action='store_true')
    parser.add_argument('--max_steps', default=150000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('--num_dataloaders', default=8, type=int)
    parser.add_argument('--max_val_examples', default=1024, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--plan_lambda', default=1.0, type=float)
    parser.add_argument('--pretrained_path', default=None, help='Path to a pre-trained TransformerSummarizer model.')
    parser.add_argument('-add_sent_toks', default=False, action='store_true')
    parser.add_argument('-fragments', default=False, action='store_true')
    parser.add_argument(
        '--summary_style',
        default='plan_abstract',
        choices=[
            'plan_abstract',
            'abstract_plan',
            'extract',
            'plan',
            'abstract',
            'hybrid_control',
        ], help='Target output during training. plan is a sequence of <s{idx}> tokens, extract is oracle summary, '
                'abstract is original reference'
    )
    parser.add_argument(
        '--oracle_cutoff', default=0.4, type=float,
        help='For summary_style=hybrid_control, summaries with ranking above this will be trained as extracts'
             '(to generate the oracle extractive summary).  Below, abstracts (to generate original reference). '
    )
    parser.add_argument('--lr', type=float, default=2.2e-4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--target_batch_size', type=int, default=16)
    parser.add_argument('--per_device_train_bs', type=int, default=8)
    parser.add_argument('--per_device_eval_bs', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'Yale-LILY/brio-cnndm-uncased',
    ])

    args = parser.parse_args()

    # Override: If we are generating a sentence plan, we MUST include <s{idx}> tokens in the source input
    args.add_sent_toks = args.add_sent_toks or args.summary_style in {'plan_abstract', 'plan', 'abstract_plan'}
    args.weight_dir = os.path.join(args.data_dir, 'weights')
    os.makedirs(args.weight_dir, exist_ok=True)

    # Set same random seed for each run
    set_same_seed(args.seed)
    run(args)
