import pandas as pd
from scipy.stats import pearsonr


if __name__ == '__main__':
    small = pd.read_csv('/nlp/projects/faithsum/results/samsum_extract_generator/test_from_beam_16_extract_4.csv')
    big = pd.read_csv('/nlp/projects/faithsum/results/samsum_extract_generator/test_from_beam_16_extract.csv')

    big['source_len'] = big['source'].apply(lambda x: len(x.split(' ')))
    small['source_len'] = small['source'].apply(lambda x: len(x.split(' ')))
    source_lens = small['source_len']

    big_wins = big['best_from_extract_rouge1_f1'] - small['best_from_extract_rouge1_f1']

    print(pearsonr(big_wins, source_lens))
    print(pearsonr(big_wins, big.diversity - small.diversity))
    print(pearsonr(big_wins, big.extract_diversity))
    print(pearsonr(source_lens, big.diversity))

    avg_len = source_lens.mean()

    short_small = small[small['source_len'] < small['source_len']]
    short_large = small[small['source_len'] >= small['source_len']]
    big_small = big[big['source_len'] < big['source_len']]
    big_large = big[big['source_len'] >= big['source_len']]

    print(short_small.best_from_extract_rouge1_f1.mean(), short_large.best_from_extract_rouge1_f1.mean())
    print(big_small.best_from_extract_rouge1_f1.mean(), big_large.best_from_extract_rouge1_f1.mean())
