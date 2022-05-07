import pandas as pd
import numpy as np


if __name__ == '__main__':
    fn = '/nlp/projects/faithsum/results/score_abstract_lower_lr/validation_beam_outputs.csv'
    df = pd.read_csv(fn)

    df['extract_over_abstract'] = df['extract_mean_f1'] >= df['mean_f1']
    df['extract_over_implied'] = df['extract_mean_f1'] >= df['implied_mean_f1']
    df['avg_diff_extract_abstract'] = abs(df['extract_mean_f1'] - df['mean_f1'])
    df['avg_diff_extract_implied'] = abs(df['extract_mean_f1'] - df['implied_mean_f1'])
    n = len(df)
    print(df['rouge1_f1'].mean(), df['rouge2_f1'].mean(), df['rougeL_f1'].mean())
    print(df['extract_rouge1_f1'].mean(), df['extract_rouge2_f1'].mean(), df['extract_rougeL_f1'].mean())
    print(df['implied_rouge1_f1'].mean(), df['implied_rouge2_f1'].mean(), df['implied_rougeL_f1'].mean())
    frac_ex_over_abs = df['extract_over_abstract'].sum() / n
    frac_ex_over_implied = df['extract_over_implied'].sum() / n

    print(frac_ex_over_abs, frac_ex_over_implied)

    df = df.assign(
        source_len=df['source'].apply(lambda x: len(x.split(' '))),
        ref_len=df['reference'].apply(lambda x: len(x.split(' ')))
    )

    print(df['avg_diff_extract_implied'].mean())
    print(df['avg_diff_extract_abstract'].mean())

    df = df.assign(
        implied_sent_pos_avg=df['implied_extract_idx'].apply(lambda x: np.mean([float(y) for y in x.split(',')])),
        implied_sent_pos_std=df['implied_extract_idx'].apply(lambda x: np.std([float(y) for y in x.split(',')])),
        extract_sent_pos_avg=df['extract_idx'].apply(lambda x: np.mean([float(y) for y in x.split(',')])),
        extract_sent_pos_std=df['extract_idx'].apply(lambda x: np.std([float(y) for y in x.split(',')]))
    )

    print(df['extract_sent_pos_avg'].mean(), df['extract_sent_pos_std'].mean())
    print(df['implied_sent_pos_avg'].mean(), df['implied_sent_pos_std'].mean())

    extract_better = df[df['extract_over_abstract']]
    abstract_better = df[~df['extract_over_abstract']]

    print(extract_better['source_len'].mean(), extract_better['ref_len'].mean())
    print(abstract_better['source_len'].mean(), abstract_better['ref_len'].mean())
