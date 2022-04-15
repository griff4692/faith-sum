# faith-sum
Effectively Faithful Summarization

```
pip install -e .
export ROUGE_HOME="{your-desired-path}/ROUGE-1.5.5/")
bash setup_rouge.sh
```

# Preprocess

```angular2html
python extract_oracles.py --dataset {cnn_dailymail,xsum} --data_dir {your-desired-data-dir} --splits train,validation,test
```

# Train Models

```angular2html
cd gen_transformers
python main.py --experiment {your-experiment} --dataset {cnn_dailymail,xsum} --summary_style {abstract,extract,abstract_plan,plan_abstract,hybrid_control} ...
```

# Evaluate

```angular2html
cd gen_transformers
python generate.py --wandb_name {your-experiment} --dataset {cnn_dailymail,xsum} --summary_style {abstract,extract,abstract_plan,plan_abstract,hybrid_control} ...
```