import csv
import parser
import os

import torch
from tqdm import tqdm
from nlp import load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset

from run_no_trainer import summarization_name_mapping


DEFAULT_KWARGS = {
    'xsum': dict(num_beams=6, length_penalty=1.0, max_length=60, min_length=10, no_repeat_ngram_size=3),
    'cnn_dailymail': dict(num_beams=4, length_penalty=2.0, max_length=140, min_length=55, no_repeat_ngram_size=3)
}


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, decoded_preds, decoded_labels):
    # preds, labels = eval_preds
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    result = metric.compute(
        predictions=decoded_preds, rouge_types=rouge_types, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gen Args')
    parser.add_argument('--hf_model', default='google/pegasus-cnn_dailymail')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_enc_length', default=1024, type=int)
    parser.add_argument('--dataset_name', default='cnn_dailymail')
    parser.add_argument('--dataset_config_name', default='3.0.0')
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--out_dir', default='/nlp/projects/faithsum')

    args = parser.parse_args()

    source_col, target_col = summarization_name_mapping[args.dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.hf_model).to(args.device).eval()

    # Dataset
    if args.dataset_name == 'cnn_dailymail':
        dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        dataset = load_dataset('xsum')

    out_dir = os.path.join(args.out_dir, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Metric
    metric = load_metric('rouge')

    for split in ['test']:
        sources = []
        targets = []
        split_data = dataset[split]
        if args.max_examples is not None and args.max_examples < len(split_data):
            rand_idxs = np.sort(np.random.choice(range(len(dataset[split])), size=args.max_examples, replace=False))
            print(f'Sampling {args.max_examples} examples')
            split_data = dataset[split].select(rand_idxs)
        out_fn = os.path.join(out_dir, f'{split}.out')
        with open(out_fn, 'w') as out:
            for ex in split_data:
                text = ex[source_col].strip()
                sources.append(text)
                targets.append(ex[target_col].strip())
            source_batches = list(chunks(sources, args.batch_size))
            target_batches = list(chunks(targets, args.batch_size))

            metrics = []
            for source_batch, target_batch in tqdm(zip(source_batches, target_batches), total=len(source_batches)):
                inputs = tokenizer(
                    source_batch, max_length=args.max_enc_length, truncation=True, padding='max_length',
                    return_tensors='pt'
                )
                with torch.no_grad():
                    summaries = model.generate(
                        input_ids=inputs['input_ids'].to(args.device),
                        attention_mask=inputs['attention_mask'].to(args.device),
                        **DEFAULT_KWARGS[args.dataset_name]
                    )

                decoded_summaries = [
                    tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries
                ]
                decoded_summaries = [d.replace('<n>', ' ') for d in decoded_summaries]
                for summary in decoded_summaries:
                    out.write(summary.replace('\n', ' ').strip() + '\n')
                metrics.append(compute_metrics(summaries.detach().cpu().numpy(), decoded_summaries, target_batch))
        metrics = pd.DataFrame(metrics)
        for col in list(metrics.columns):
            print(f'{col}: {round(metrics[col].dropna().mean(), 3)}')
        metrics_fn = os.path.join(out_dir, f'{split}.csv')
        metrics.to_csv(metrics_fn, index=False)
