import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import AutoTokenizer, BartTokenizer

from gen_transformers.dataset import SummaryDataModule
from gen_transformers.model import TransformerSummarizer
from global_utils import get_free_gpus, set_same_seed
from gen_transformers.model_utils import infer_hf_model
from gen_transformers.data_utils import infer_dataset


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
    if 'pegasus' in args.hf_model:
        print('Using full precision for PEGASUS.')
        precision = 32

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
        val_check_interval = 1.0 if args.debug else 0.25
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

        model.save_hyperparameters(args)
        val_check_interval = 1.0 if args.debug else 0.25

    datamodule = SummaryDataModule(args, tokenizer=tokenizer)

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='faith_sum',
        entity='griffinadams',
    )

    monitor_metric = f'validation/{args.val_monitor_metric}'
    mode = args.val_metric_mode

    callbacks = []
    if args.save_top_k < 1:
        print('Not saving checkpoints. Will have to re-run.')
        checkpoint_callback = None
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            save_top_k=args.save_top_k,
            save_last=args.save_top_k > 1,
            mode=mode
        )
        callbacks.append(checkpoint_callback)
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
        num_sanity_val_steps=1 if args.debug else 2,
        log_every_n_steps=25,
        max_steps=args.max_steps,
        plugins=plugins,
        # detect_anomaly=args.debug
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-4, max_lr=1e-2, update_attr=True, num_training=100)
        print(lr_finder.results)
    else:
        print('Starting training...')
        trainer.fit(model, datamodule=datamodule)
        if checkpoint_callback is not None:
            print(f'Best weights saved --> {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BART/PEGASUS trainer.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--restore_path', default=None)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_gpus', default=1, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--max_val_examples', default=1024, type=int)
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('-no_schedule', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--num_dataloaders', default=8, type=int)
    parser.add_argument('-extract_indicators', default=False, action='store_true')
    parser.add_argument('--unlike_coef', default=1.0, type=float)
    parser.add_argument('--corrupt_strategy', default='swap')
    parser.add_argument('--copy_bart_class_dropout', default=0.1, type=float)
    parser.add_argument('-add_brio_loss', default=False, action='store_true')
    parser.add_argument('-use_oracle_candidates', default=False, action='store_true')
    parser.add_argument('-is_word_brio', default=False, action='store_true')
    # parser.add_argument('-use_regular_candidates', default=False, action='store_true')
    # parser.add_argument('--brio_experiment', default='add_doc')
    parser.add_argument('--brio_weight', default=1, type=float)
    parser.add_argument('--mle_weight', default=1, type=float)
    parser.add_argument('--brio_margin', default=0.001, type=float)
    parser.add_argument('--brio_length_penalty', default=2.0, type=float)
    parser.add_argument('--brio_scale', default=1.0, type=float)
    parser.add_argument('--max_brio_candidates', default=10, type=int)
    parser.add_argument('--brio_score_mode', default='likelihood', choices=['score', 'likelihood', 'similarity'])
    parser.add_argument('-include_gold', default=False, action='store_true')

    parser.add_argument('--val_monitor_metric', default='combined')
    parser.add_argument('--val_metric_mode', default='min')
    parser.add_argument('--oracle_drop_p', default=0.0, type=float)
    parser.add_argument('--oracle_column', default='oracle_idxs', choices=['oracle_idxs', 'oracle_idxs_bert'])

    # Hyper-Parameters
    parser.add_argument('--lr', type=float, default=1e-5)  # used to be 3e-5
    parser.add_argument('--high_lr', type=float, default=1e-3)  # For newly initialized parameters
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
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('-skip_if_present', default=False, action='store_true')
    parser.add_argument('--extract_method', type=str, default='generate', choices=['generate', 'select'])
    parser.add_argument('--pretrained_path', default=None, help='Path to a pre-trained TransformerSummarizer model.')
    # HuggingFace identifier of model for which to load weights for fine-tuning
    parser.add_argument('--hf_model', default=None)

    # Task-specific / Project-specific parameters
    parser.add_argument(
        '--summary_style',
        default='extract',
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

    infer_dataset(args, 'experiment')
    infer_hf_model(args)

    # Won't held yet for multi-gpu
    args.grad_accum = args.target_batch_size // args.per_device_train_bs

    # if args.debug:  # Use small data and tiny BART model
    #     args.hf_model = 'sshleifer/bart-tiny-random'

    # Override: If we are generating an extract, we MUST include <s{idx}> tokens in the source input
    args.add_sent_toks = args.add_sent_toks or 'extract' in args.summary_style or args.extract_indicators
    if args.add_sent_toks:
        print('Pre-pending each sentence in the source document with special token <s{idx}>.')

    args.weight_dir = os.path.join(args.data_dir, 'weights')
    experiment_dir = os.path.join(args.weight_dir, args.experiment)

    if os.path.exists(experiment_dir) and args.skip_if_present:
        print(f'{experiment_dir} already exists.  Skipping.  Remove -skip_if_present to re-run.')
        exit(0)
    print(f'Setting up {args.weight_dir} to store model weights, metrics, and results.')
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable

    # Set same random seed for each run
    set_same_seed(args.seed)
    run(args)
