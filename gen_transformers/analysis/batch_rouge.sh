#!/bin/bash

python add_eval_rouge.py --fn /nlp/projects/faithsum/results/gen_abstract_full/test_nucleus_16_outputs.csv --columns abstract,implied_extract
python add_eval_rouge.py --fn /nlp/projects/faithsum/results/bart_large_cnn/test_nucleus_16_outputs.csv --columns abstract,implied_extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc/test_beam_16_outputs.csv --columns extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc/test_diverse_16_outputs.csv --columns extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/add_doc/test_beam_16_outputs.csv --columns extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/bart_large_cnn/test_beam_16_outputs.csv --columns abstract,implied_extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/bart_large_cnn/test_diverse_16_outputs.csv --columns abstract,implied_extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/gen_abstract_full/test_beam_16_outputs.csv --columns abstract,implied_extract
#python add_eval_rouge.py --fn /nlp/projects/faithsum/results/gen_abstract_full/test_diverse_16_outputs.csv --columns abstract,implied_extract
