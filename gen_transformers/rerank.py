import os

import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm
from datasets import load_from_disk
from gen_transformers.dataset import get_sent_ngrams
from transformers import AutoTokenizer, BartTokenizer

os.environ['ROUGE_HOME'] = os.path.expanduser('~/faith-sum/eval/ROUGE-1.5.5/')
from data_utils import get_path_from_exp
from gen_transformers.model import TransformerSummarizer
from gen_transformers.gen_from_extract import get_extract_idxs_from_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking Scores for Inference.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--gpu_device', default=1, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='brio_score_w_doc')
    parser.add_argument('--rank_experiment', default='gen_extract_full_ar_mask_red_feat')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--split', default='validation')
    parser.add_argument('--hf_model', default='facebook/bart-base')
    parser.add_argument('--max_examples', default=999999, type=int)
    parser.add_argument('--score_mode', default='score')

    # Hyper-parameters: Should be the same as used when training
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_num_sents', type=int, default=200)

    args = parser.parse_args()
    weight_dir = os.path.join(args.data_dir, 'weights')

    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
    tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
    print(f'Loading tokenizer from {tokenizer_dir}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)
    except:  # BRIO model doesn't load from AutoTokenizer
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

    print(f'Loading model from {ckpt_path}...')
    model = TransformerSummarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
    ).to(args.gpu_device).eval()

    results_dir = os.path.join(args.data_dir, 'results', args.rank_experiment)
    rank_fn = os.path.join(results_dir, f'{args.split}_sample_outputs.csv')
    print(f'Loading in predictions from {rank_fn}')
    outputs = pd.read_csv(os.path.join(results_dir, f'{args.split}_diverse_sample_outputs.csv'))
    outputs.dropna(subset=['extract'], inplace=True)
    n = len(outputs)
    if n > args.max_examples:
        outputs = outputs.sample(n=args.max_examples, replace=False, random_state=111)
        n = len(outputs)

    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = load_from_disk(data_dir)[args.split]
    dataset_idx2id = dataset['id']
    all_source_annotated = dataset['source_annotated']
    all_input_ids = dataset['input_ids']
    records = outputs.to_dict('records')
    special_id_min = min(tokenizer.additional_special_tokens_ids)
    corels = []
    top_ranks = []
    pred_rouges = []
    first_rouges = []

    for record in tqdm(records, total=len(records)):
        source_annotated = all_source_annotated[record['dataset_idx']]
        source_ngrams = get_sent_ngrams(source_annotated)
        input_ids = all_input_ids[record['dataset_idx']]
        # Get source tokens
        reference = record['reference']
        extract_rouges = np.array([float(x) for x in record['extract_rouges'].split(',')])
        extract_idx = [get_extract_idxs_from_str(x) for x in record['extract_idx'].split('<cand>')]

        encoder_inputs = {
            'input_ids': torch.LongTensor(input_ids).to(args.gpu_device).unsqueeze(0)
        }
        with torch.no_grad():
            encoder_outputs = model.get_encoder_h(encoder_inputs)
        encoder_h = encoder_outputs.last_hidden_state
        cls_mask = encoder_inputs['input_ids'] >= special_id_min
        cls_mask[:, 0] = True  # Doc Token

        cls_h = encoder_h[cls_mask].unsqueeze(0)
        num_cand = len(extract_idx)

        eos_token_id = cls_h.size()[1] - 1  # Decrement the document token
        if args.score_mode == 'likelihood':
            # Add 'dynamic' eos token id to end of sequences
            for i in range(num_cand):
                extract_idx[i].append(eos_token_id)

            max_len = max([len(x) for x in extract_idx])

            brio_labels = np.zeros([num_cand, max_len], dtype=np.int64)
            brio_labels.fill(-100)
            for cand_idx in range(num_cand):
                num = len(extract_idx[cand_idx])
                brio_labels[cand_idx, :num] = extract_idx[cand_idx]
            brio_labels = torch.from_numpy(brio_labels).to(args.gpu_device)
            model.sent_bart.source_ngrams_cache = source_ngrams
        else:
            brio_labels = None  # Encoder-only

        stop_input_id = torch.LongTensor([0]).to(args.gpu_device)
        inputs_embeds = torch.cat([cls_h, model.stop_embed(stop_input_id).unsqueeze(0)], dim=1)
        if args.score_mode == 'likelihood':
            inputs_embeds = inputs_embeds.repeat(num_cand, 1, 1)
            contrast_outputs = model.sent_bart(inputs_embeds=inputs_embeds, calculate_loss=False, labels=brio_labels)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            nll = loss_fct(contrast_outputs.logits.view(-1, eos_token_id), brio_labels.view(-1)).view(num_cand, -1)
            seq_lens = (brio_labels > -100).sum(dim=1) ** args.brio_length_penalty
            scores = (- nll.sum(dim=1) / seq_lens).detach().cpu().numpy().tolist()
        else:
            with torch.no_grad():
                encoder_h = model.sent_bart.model.encoder(inputs_embeds=inputs_embeds)[0][0]
            encoder_sent = encoder_h[1:, :]
            pooled_extract = torch.stack([encoder_sent[x].mean(dim=0) for x in extract_idx])
            encoder_doc_rep = encoder_h[:1, :].repeat(len(pooled_extract), 1)
            pooled = torch.cat([encoder_doc_rep, pooled_extract], dim=1)
            scores = model.sent_bart.calibration_classifier(pooled).squeeze(1).detach().cpu().numpy().tolist()

        corels.append(spearmanr(scores, extract_rouges)[0])
        pred_priority = np.argsort(-np.array(scores))
        pred_ordered_rouges = [extract_rouges[pred_idx] for pred_idx in pred_priority]
        top_ranks.append(int(np.argmax(pred_ordered_rouges)) + 1)
        pred_rouges.append(pred_ordered_rouges[0])
        first_rouges.append(extract_rouges[0])

        print(np.mean(corels), np.mean(top_ranks), np.mean(pred_rouges), np.mean(first_rouges))
