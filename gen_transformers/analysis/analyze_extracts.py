import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/faithsum/results/cnn_e_v1/test_beam_16_outputs.csv')

    rouges_by_beam = [
        [] for _ in range(16)
    ]
    lens_by_beam = [
        [] for _ in range(16)
    ]

    for rouges in df['extract_rouges']:
        for beam, score in enumerate(rouges.split(',')):
            rouges_by_beam[beam].append(float(score))

    for extract in df['extract_idx']:
        for beam, idxs in enumerate(extract.split('<cand>')):
            lens_by_beam[beam].append(len(idxs.split(',')))

    print('Beam,Rouge,Length')
    for beam in range(16):
        print(beam, np.mean(rouges_by_beam[beam]), np.mean(lens_by_beam[beam]))
