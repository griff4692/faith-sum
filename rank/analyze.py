import pandas as pd
from scipy.stats import pearsonr
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv(
        '/nlp/projects/faithsum/results/gen_extract_full_ar_mask_red_feat/validation_sample_outputs_with_gen_extract_sample_feats_ranking.csv'
    )

    records = df.to_dict('records')
    corels = []
    for record in records:
        try:
            preds = [float(x) for x in record['rank_pred'].split(',')]
            rouges = [float(x) for x in record['extract_rouges'].split(',')]
            corels.append(pearsonr(preds, rouges)[0])
        except:
            print('Could not process')

    print(np.mean(corels))
