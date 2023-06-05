# Generating EDU Extracts for Plan-Guided Summary Re-Ranking

ACL 2023 [Long Paper](https://arxiv.org/abs/2305.17779).

# Setup

```
pip install -e .
cd transformers && pip install -e . && cd ../
export DATA_DIR={path for saving data/models/results}
export ROUGE_HOME="{your-desired-path}/ROUGE-1.5.5/"
bash setup_rouge.sh
python -m spacy download en_core_web_sm
export WANDB_API_KEY={a weights & biases API key}
```


If you want to use BRIO re-ranker, please feel README under `./BRIO`. The directory is [borrowed](https://github.com/yixinL7/BRIO) from the original [BRIO paper](https://arxiv.org/abs/2203.16804).

# Preprocess

```angular2html
python preprocess/split_into_sentences.py
[ Run EDU script ]
python preprocess/add_edu.py
python preprocess/align_edu.py
```

# Training

## EDU Extract Generator

```angular2html
python gen_transformers/main.py --dataset cnn_dailymail --summary_style extract --experiment {my-extract-experiment} --wandb_project {yours} --wandb_entity {yours}
```

## Extract-Guided Abstractor

```angular2html
python gen_transformers/main.py --summary_style abstract -extract_indicators --dataset cnn_dailymail --mle_weight 1.0 --like_coef 1.0 --unlike_coef 1.0 --experiment {my-abstract-experiment} --wandb_project {yours} --wandb_entity {yours}
```

See [paper](https://arxiv.org/abs/2305.17779) for guidance on setting Hyper-Parameters (`like_coef`, `unlike_coef`, and `mle_weight`). Setting to all `1` will give you reliable performance, and results are not overly sensitive to changes in the coefficients.

# Generating Plan-Guided Abstracts

```angular2html
bash gen_transformers/ea_pipeline.sh {device} {validation,test} {my-extract-experiment} {my-abstract_experiment} {max_examples}
```

EA stands for extract-abstract and performs the two-step generation described in the paper. Manually change `N_CAND=16` in `ea_pipeline.sh` to generate a different number of candidates.

## Prepare for BRIO re-ranking
