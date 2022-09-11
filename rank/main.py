import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import AutoTokenizer
from global_utils import get_free_gpus, set_same_seed
from rank.dataset import RankDataModule
from rank.model import SummaryRanker


def build_tokenizer(max_num_sents=200):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    add_tokens = [f'<s{i}>' for i in range(max_num_sents)]
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


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
    print('Num GPUs --> {}'.format(args.num_gpus))
    precision = 16 if args.num_gpus is not None else 32
    experiment_dir = os.path.join(args.weight_dir, args.experiment, 'rank')
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable

    tokenizer = build_tokenizer(max_num_sents=args.max_num_sents)
    model = SummaryRanker(args, tokenizer)
    datamodule = RankDataModule(args, tokenizer=tokenizer)

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='rerank',
        entity='griffinadams',
    )

    primary_eval_metric = 'val/score'
    primary_metric_mode = 'max'  # Higher is better ('min' for val_loss)
    checkpoint_callback = ModelCheckpoint(
        monitor=primary_eval_metric,
        save_top_k=1,
        save_last=False,
        mode=primary_metric_mode
    )
    # early_stopping = EarlyStopping(primary_eval_metric, mode=primary_metric_mode, patience=5, verbose=True)
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
        val_check_interval=1.0 if args.debug else 0.1,
        num_sanity_val_steps=0 if args.debug else 2,
        log_every_n_steps=10,
        max_steps=args.max_steps,
        plugins=plugins,
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-4, max_lr=1e-2, update_attr=True, num_training=100)
        print(lr_finder.results)
    else:
        print('Starting training...')
        trainer.fit(model, datamodule=datamodule)
        print(f'Best weights saved --> {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summarization Re-Ranker.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--restore_path', default=None)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_gpus', default=1, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--max_val_examples', default=1024, type=int)
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--gen_experiment', default='gen_extract_full_ar_mask_red_feat')
    parser.add_argument('-no_schedule', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--num_dataloaders', default=8, type=int)

    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_num_sents', type=int, default=200)

    args = parser.parse_args()

    args.weight_dir = os.path.join(args.data_dir, 'weights')
    os.makedirs(args.weight_dir, exist_ok=True)

    # Set same random seed for each run
    set_same_seed(args.seed)
    run(args)
