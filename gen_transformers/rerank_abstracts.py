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


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return np.array([float(y) for y in num_str.split(delim)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking Scores for Abstract Inference.')

    # Configuration Parameters
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--gpu_device', default=1, type=int)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--wandb_name', default='from_extract_rerank_bart_large_cnn')
    parser.add_argument('--prediction_col', default='from_extract_abstract')
    parser.add_argument('--rank_experiment', default='add_doc')
    # How many processes to use when loading batches on CPU
    parser.add_argument('--split', default='validation')
    parser.add_argument('--hf_model', default='facebook/bart-large-cnn')
    parser.add_argument('--max_examples', default=999999, type=int)
    parser.add_argument('--score_mode', default='likelihood')

    # Hyper-parameters: Should be the same as used when training
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_num_sents', type=int, default=200)

    args = parser.parse_args()
    weight_dir = os.path.join(args.data_dir, 'weights')

    rouge_col = (
        'eval_from_extract_abstract_rouge1_f1'
        if args.prediction_col == 'from_extract_abstract' else 'eval_rouge1_f1'
    )

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
    rank_fn = os.path.join(results_dir, f'{args.split}_from_sample_w_diverse_extract_4.csv')
    print(f'Loading in predictions from {rank_fn}')
    outputs = pd.read_csv(rank_fn)
    outputs.dropna(subset=[args.prediction_col], inplace=True)
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
    corels = []
    top_ranks = []
    pred_rouges = []
    first_rouges = []

    for record in tqdm(records, total=len(records)):
        source_annotated = all_source_annotated[record['dataset_idx']]
        input_ids = all_input_ids[record['dataset_idx']]
        min_sent_id = input_ids[1]
        input_ids = [x for x in input_ids if x < min_sent_id]

        # Get source tokens
        reference = record['reference']

        rouges = get_arr(record[rouge_col])
        candidates = record['from_extract_abstract'].split('<cand>')

        encoder_inputs = {
            'input_ids': torch.LongTensor(input_ids).to(args.gpu_device).unsqueeze(0)
        }
        with torch.no_grad():
            encoder_outputs = model.get_encoder_h(encoder_inputs)
        encoder_h = encoder_outputs.last_hidden_state

        # from_extract_abstracts
        with tokenizer.as_target_tokenizer():
            brio_labels = np.array(tokenizer(
                candidates, max_length=1024, truncation=True, padding='longest'
            )['input_ids'], dtype=np.int64)
            brio_labels[np.where(brio_labels == tokenizer.pad_token_id)] = -100
        brio_labels = torch.from_numpy(brio_labels).to(args.gpu_device)
        num_cand, target_len = brio_labels.size()

        stop_input_id = torch.LongTensor([0]).to(args.gpu_device)
        encoder_outputs_rep = encoder_h.repeat(num_cand, 1, 1).contiguous()
        inputs = {
            'encoder_outputs': [encoder_outputs_rep],
            'labels': brio_labels,
        }
        contrast_outputs = model.model(**inputs, use_cache=False, output_hidden_states=True)

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = loss_fct(
            contrast_outputs.logits.view(-1, contrast_outputs.logits.size()[-1]), brio_labels.view(-1)
        ).view(num_cand, target_len)
        seq_lens = (brio_labels > -100).sum(dim=1) ** model.hparams.brio_length_penalty
        scores = ((- nll.sum(dim=1) / seq_lens) * model.hparams.brio_scale).detach().cpu().numpy().tolist()

        corel = spearmanr(scores, rouges)[0]
        if min(scores) == max(scores):
            print('All predicted equal')

        if np.isnan(corel):
            print('NaN correlation')
        else:
            corels.append(corel)

        pred_priority = np.argsort(-np.array(scores))
        pred_ordered_rouges = [rouges[pred_idx] for pred_idx in pred_priority]
        top_ranks.append(int(np.argmax(pred_ordered_rouges)) + 1)
        pred_rouges.append(pred_ordered_rouges[0])
        first_rouges.append(rouges[0])
        print(f'Corel: {np.mean(corels)}. Rank: {np.mean(top_ranks)}, Predicted ROUGE-1: {np.mean(pred_rouges)}. '
              f'First Candidate ROUGE-1: {np.mean(first_rouges)}')
        record['rank_scores'] = ','.join(list(map(str, scores)))
    outputs = pd.DataFrame(records)
    print('Fini!')
    print(
        f'Corel: {np.mean(corels)}. Rank: {np.mean(top_ranks)}, Predicted ROUGE-1: {np.mean(pred_rouges)}. '
        f'First Candidate ROUGE-1: {np.mean(first_rouges)}')
    rank_out_fn = rank_fn.split('.')[0] + '_w_predicted_ranks.csv'
    print(f'Saving back to {rank_out_fn}')
    outputs.to_csv(rank_out_fn, index=False)
