import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    fn = '/nlp/projects/faithsum/results/score_abstract_v2/validation_beam_outputs.csv'
    df = pd.read_csv(fn)

    df['extract_over_abstract'] = df['extract_mean_f1'] >= df['mean_f1']
    df['extract_over_implied'] = df['extract_mean_f1'] >= df['implied_mean_f1']
    df['avg_diff_extract_abstract'] = abs(df['extract_mean_f1'] - df['mean_f1'])
    df['avg_diff_extract_implied'] = abs(df['extract_mean_f1'] - df['implied_mean_f1'])
    n = len(df)
    print('Abstract ROUGE F1: ', df['rouge1_f1'].mean(), df['rouge2_f1'].mean(), df['rougeL_f1'].mean())
    print('Extract ROUGE F1: ', df['extract_rouge1_f1'].mean(), df['extract_rouge2_f1'].mean(), df['extract_rougeL_f1'].mean())
    print('Implied ROUGE F1: ', df['implied_rouge1_f1'].mean(), df['implied_rouge2_f1'].mean(), df['implied_rougeL_f1'].mean())
    frac_ex_over_abs = df['extract_over_abstract'].sum() / n
    frac_ex_over_implied = df['extract_over_implied'].sum() / n

    print(f'Fraction Extractive Better than Abstractive / Implied: ', frac_ex_over_abs, frac_ex_over_implied)

    df = df.assign(
        source_len=df['source'].apply(lambda x: len(x.split(' '))),
        ref_len=df['reference'].apply(lambda x: len(x.split(' ')))
    )

    print('Average Difference in mean ROUGE between extract and implied: ', df['avg_diff_extract_implied'].mean())
    print('Average Difference in mean ROUGE between extract and abstract: ', df['avg_diff_extract_abstract'].mean())

    df = df.assign(
        implied_sent_pos_avg=df['implied_extract_idx'].apply(lambda x: np.mean([float(y) for y in x.split(',')])),
        implied_sent_pos_std=df['implied_extract_idx'].apply(lambda x: np.std([float(y) for y in x.split(',')])),
        extract_sent_pos_avg=df['extract_idx'].apply(lambda x: np.mean([float(y) for y in x.split(',')])),
        extract_sent_pos_std=df['extract_idx'].apply(lambda x: np.std([float(y) for y in x.split(',')]))
    )

    print('Extract Sent Position: ', df['extract_sent_pos_avg'].mean(), df['extract_sent_pos_std'].mean())
    print('Implied Sent Position: ', df['implied_sent_pos_avg'].mean(), df['implied_sent_pos_std'].mean())

    extract_better = df[df['extract_over_abstract']]
    abstract_better = df[~df['extract_over_abstract']]

    print('Source / Reference Length: ', df['source_len'].mean(), df['ref_len'].mean())
    print('Extract Outperforms when Source / Reference Length: ', extract_better['source_len'].mean(), extract_better['ref_len'].mean())
    print('Abstract Outperforms when Source / Reference Length: ', abstract_better['source_len'].mean(), abstract_better['ref_len'].mean())

    df = df.assign(
        source_bin=pd.qcut(df['source_len'], 4, labels=['Shortest', 'Short', 'Long', 'Longest'])
    )

    for col in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']:
        df['abstract_' + col] = df[col]

    bounds = {
        '1': [0.39, 0.46],
        '2': [0.15, 0.24],
        'L': [0.32, 0.44]
    }
    for r, bound in bounds.items():
        plt.close()
        df_plot = df[['source_bin', f'abstract_rouge{r}_f1', f'extract_rouge{r}_f1', f'implied_rouge{r}_f1']]
        tuna = df_plot.melt('source_bin', var_name='Summary Type', value_name=f'ROUGE{r}-F1')
        print(f'Building lineplot for ROUGE-{r}')
        ax = sns.barplot(data=tuna, x='source_bin', hue='Summary Type', y=f'ROUGE{r}-F1', palette='Blues_d', ci=None)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        # min_val = tuna[f'ROUGE{r}-F1'].min()
        # max_val = tuna[f'ROUGE{r}-F1'].max()
        ax.set_ylim(bound[0], bound[1])
        fig_fn = f'rouge{r}_by_length.png'
        print(f'Saving figure to {fig_fn}')
        plt.savefig(fig_fn, bbox_inches='tight')
