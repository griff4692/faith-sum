from datasets import load_from_disk
import regex as re

from preprocess.align_edu import edus_from_html
from gen_transformers.gen_from_extract import compute_rouge
from eval.rouge_metric import RougeMetric


if __name__ == '__main__':
    DATASET = 'cnn_dailymail'
    data_dir = f'/nlp/projects/faithsum/{DATASET}_edu_alignments'
    data = load_from_disk(data_dir)
    val = data['validation']

    print(val[0].keys())

    rouge_metric = RougeMetric()

    rouges = []
    for example in val:
        source_edus = edus_from_html(example['source_edu_annotated'])
        reference = example['highlights']
        oracle_edus = [source_edus[i] for i in example['oracle_idxs']]
        oracle_edus_flat = ' '.join(oracle_edus)

        rouge = compute_rouge(oracle_edus_flat, reference, rouge_metric=rouge_metric)

        rouges.append(rouge)
