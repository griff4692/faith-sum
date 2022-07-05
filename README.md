# faith-sum

# Setup

```
pip install -e .
cd transformers && pip install -e . && cd ../
export ROUGE_HOME="{your-desired-path}/ROUGE-1.5.5/")
bash setup_rouge.sh
python -m spacy download en_core_web_sm
export WANDB_API_KEY={your weights & biases API key - ask Griffin if you don't have one}
```

# Preprocess

```angular2html
python preprocess/truncate_dataset.py --dataset {cnn_dailymail,xsum} --data_dir {your-desired-data-dir} --splits train,validation,test --hf_model google/pegasus-large
```

# Train Models

```angular2html
cd gen_transformers
python main.py --experiment {your-experiment} --dataset {cnn_dailymail,xsum} --summary_style {abstract,extract,abstract_plan,plan_abstract,hybrid_control} --hf_model google/pegasus-large ...
```

# Evaluate

```angular2html
cd gen_transformers
python generate.py --wandb_name {your-experiment} --dataset {cnn_dailymail,xsum} --summary_style {abstract,extract,abstract_extract} ...
```