import os.path
import shutil

import numpy as np
from preprocess.align_edu import edus_from_html

from datasets import load_from_disk


if __name__ == '__main__':
    experiments = [
        # 'cnn_e_v1_ea_rand_v2', 'bart_large_cnn_beam', 'bart_large_cnn_dbs_dendrite',
        # 'bart_large_cnn_nucleus_v2'
        'cnn_e_final_cnn_ea_final_mle_10.0_like_1_unlike_1'
    ]

    from_dirs = []
    to_dirs = []
    import os
    for experiment in experiments:
        from_dirs.append(os.path.expanduser(f'~/BRIO/result/{experiment}'))
        x = []
        for quart in [1, 2, 3, 4]:
            x.append(os.path.expanduser(f'~/BRIO/result/{experiment}_{quart}_length_quartile/'))
        for z in x:
            os.makedirs(os.path.join(z, 'reference_ranking'), exist_ok=True)
            os.makedirs(os.path.join(z, 'candidate_ranking'), exist_ok=True)
        to_dirs.append(x)
    test = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')['test']

    source_edus = list(map(edus_from_html, test['source_edu_annotated']))
    source_num_edus = list(map(len, source_edus))

    quartiles = np.array_split(np.argsort(source_num_edus), 4)

    for qidx, quartile in enumerate(quartiles):
        print(f'Starting quartile {qidx}')
        quart_dataset_idxs = test.select(quartile)['dataset_idx']
        for dataset_idx in quart_dataset_idxs:
            for from_dir, to_dir in zip(from_dirs, to_dirs):
                quartile_to_dir = to_dir[qidx]
                for (prefix, suffix) in [('candidate', 'dec'), ('reference', 'ref')]:
                    from_fn = os.path.join(from_dir, f'{prefix}_ranking', f'{dataset_idx}.{suffix}')
                    to_fn = os.path.join(quartile_to_dir, f'{prefix}_ranking', f'{dataset_idx}.{suffix}')
                    shutil.copy(from_fn, to_fn)
