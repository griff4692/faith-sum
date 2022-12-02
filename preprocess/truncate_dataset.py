import os
import regex as re

import argparse
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer
from bert_score.scorer import BERTScorer
from transformers import AutoModel

from preprocess.extract_oracles import convert_to_sents
from preprocess.convert_abstractive_to_extractive import gain_selection
from sum_constants import summarization_name_mapping
from preprocess.bert_align_playground import add_bert_alignment_no_red


BS_PARAMS = {
    'samsum': {
        # 'threshold': 0.87,
        # 'p_factor': 0.8,
        # 'max_per_sent': 3,
        'avg_imp_threshold': 0.02,
        'max_imp_threshold': 0.15,
        'max_coverage': 0.95,
        'max_retrievals': 4,
    }
}


def get_ids(args, nlp, tokenizer, batch_data, input_col, target_col, bs, bs_tokenizer, max_input_length=1024, max_output_length=256):
    batch_source_sents = [
        convert_to_sents(inputs, nlp, is_dialogue=args.dataset == 'samsum') for inputs in batch_data[input_col]
    ]
    source_annotated = [
        # TODO test this
        # WARNING!  When running CNN/DM, s.text.strip() was just s
        # Might be different pre-processing results
        ''.join(
            [f'<s{i}> {s.text.strip()}' for i, s in enumerate(source_sents) if i < args.max_num_sents]
        ) for source_sents in batch_source_sents
    ]

    input_ids = tokenizer(
        source_annotated,
        truncation=True,
        max_length=max_input_length,
    )['input_ids']
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    num_sents = [
        len(re.findall('<s\d+>', x)) for x in decoded
    ]
    labels = tokenizer(
        batch_data[target_col],
        truncation=True,
        max_length=max_output_length,
    )['input_ids']

    oracle_idxs = []
    oracle_bert_idxs = []
    batch_size = len(source_annotated)
    oracle_rouge1 = []
    oracle_rouge2 = []
    rouge1_history = []
    rouge2_history = []
    best_history = []
    for batch_idx in range(batch_size):
        target = batch_data[target_col][batch_idx]
        target_sents = convert_to_sents(target, nlp, is_dialogue=False)  # Summaries never in dialogue format
        source_sents = [sent for i, sent in enumerate(batch_source_sents[batch_idx]) if i < num_sents[batch_idx]]
        source_sents_tok = [[str(token.text) for token in sentence] for sentence in source_sents]
        target_sents_tok = [[str(token.text) for token in sentence] for sentence in target_sents]
        # Sort oracle order or not
        idxs, rouge, r1_hist, r2_hist, best_hist = gain_selection(
            source_sents_tok, target_sents_tok, 5, lower=True, sort=False)

        # threshold = BS_PARAMS[args.dataset]['threshold']
        # p_factor = BS_PARAMS[args.dataset]['p_factor']
        # max_per_sent = BS_PARAMS[args.dataset]['max_per_sent']
        source_sents_str = [str(x) for x in source_sents]
        target_sents_str = [str(x) for x in target_sents]
        if bs is not None:
            try:
                # bert_idxs, _ = add_bert_alignment(
                #     bs, source_sents_str, target_sents_str, threshold=threshold, p_factor=p_factor,
                #     max_per_sent=max_per_sent
                # )

                bert_idxs, _ = add_bert_alignment_no_red(
                    bs, bs_tokenizer, source_sents_str, target_sents_str,
                    **BS_PARAMS[args.dataset]
                )
            except:
                print('Error with BertScore. Probably empty source or target. Setting to same as ROUGE gain')
                bert_idxs = idxs
            oracle_bert_idxs.append(bert_idxs)

        rouge1_history.append(r1_hist)
        rouge2_history.append(r2_hist)
        best_history.append(best_hist)
        oracle_idxs.append(idxs)
        oracle_rouge1.append(rouge['rouge_1'])
        oracle_rouge2.append(rouge['rouge_2'])

    row = {
        'source_annotated': source_annotated,
        'input_ids': input_ids,
        'labels': labels,
        'num_source_sents_pre_trunc': [len(x) for x in batch_source_sents],
        'num_source_sents': num_sents,
        'oracle_idxs': oracle_idxs,
        'oracle_rouge1': oracle_rouge1,
        'oracle_rouge2': oracle_rouge2,
        'rouge1_history': rouge1_history,
        'rouge2_history': rouge2_history,
        'best_history': best_history,
    }
    if bs is not None:
        row['oracle_idxs_bert'] = oracle_bert_idxs
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='samsum')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--hf_model', default='facebook/bart-base', choices=[
        'facebook/bart-base',
        'facebook/bart-large',
        'facebook/bart-large-cnn',
        'facebook/bart-large-xsum',
        'google/pegasus-xsum',
        'lidiya/bart-large-xsum-samsum',
    ])
    parser.add_argument('--num_proc', default=16, type=int)
    parser.add_argument('--max_num_sents', default=200, type=int)
    parser.add_argument('-add_bert', default=False, action='store_true')

    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    bs = None
    bs_tokenizer = None
    if args.add_bert:
        args.num_proc = 1  # Can't use torch in multi-process without 'spawn' start method (not worth it)
        # bs = BERTScorer(lang='en', idf=False)
        hf = 'microsoft/deberta-large-mnli'
        bs = AutoModel.from_pretrained(hf).eval().to(0)
        bs_tokenizer = AutoTokenizer.from_pretrained(hf)

    input_col, target_col = summarization_name_mapping[args.dataset]
    if 'pegasus' in args.hf_model:
        out_dir = os.path.join(args.data_dir, args.dataset + '_pegasus')
    else:
        out_dir = os.path.join(args.data_dir, args.dataset)

    if args.dataset == 'xsum':
        max_input_length = 512
        max_output_length = 64
    else:
        max_input_length = 1024
        max_output_length = 256

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    add_tokens = [f'<s{i}>' for i in range(args.max_num_sents)]
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    encoded_data = {}
    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        encoded = dataset[split].map(lambda examples: get_ids(
            args, nlp, tokenizer, examples, input_col, target_col, bs, bs_tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        ), batched=True, batch_size=100, num_proc=args.num_proc)
        encoded = encoded.filter(lambda example: len(example[input_col].strip()) > 0)
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
