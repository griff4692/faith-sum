#!/bin/bash
set -e
export ROUGE_HOME=/home/griffin/faith-sum/eval/ROUGE-1.5.5/

#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_beam_16_outputs.csv --columns extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_beam_1_outputs.csv --columns extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_diverse_16_outputs.csv --columns extract

python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_from_beam_16_extract.csv --columns extract,from_extract_abstract
python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc_bart_large_cnn/test_from_diverse_16_extract.csv --columns extract,from_extract_abstract
