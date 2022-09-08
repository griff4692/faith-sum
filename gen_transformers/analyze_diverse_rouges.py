import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/faithsum/results/gen_extract_full_ar_mask_red_feat/train_sample_outputs.csv')

    extract_rouges = [[float(y) for y in x.split(',')] for x in df['extract_rouges'].tolist()]
    n = len(df)
    scores = [0 for _ in range(len(extract_rouges[0]))]

    for rouge_arr in extract_rouges:
        for i, s in enumerate(rouge_arr):
            scores[i] += s
    for beam, score in enumerate(scores):
        print(f'Beam {beam}: {score / n}')
