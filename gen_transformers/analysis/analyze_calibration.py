import pandas as pd
import regex as re

from datasets import load_from_disk
import numpy as np

from bert_score.scorer import BERTScorer
from tqdm import tqdm

from scipy.stats import spearmanr


if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/faithsum/results/samsum_brio_score_bert_extract/test_from_beam_16_extract.csv')

    bs = BERTScorer(model_type='microsoft/deberta-large-mnli')

    corels = []
    beam_to_down = []
    cal_target_to_down = []
    cal_to_down = []
    cal_to_target = []
    dataset = load_from_disk('/nlp/projects/faithsum/samsum')['test']
    records = df.to_dict('records')
    for record in tqdm(records):
        extracts = record['extract'].split('<cand>')
        ea_rouge = [float(x) for x in record['from_extract_rouges'].split('<cand>')]
        calibration_scores = [float(x) for x in record['calibrated_beam_score'].split(',')]

        dataset_idx = record['dataset_idx']
        source_annotated = dataset[dataset_idx]['source_annotated']
        oracle_idx = dataset[dataset_idx]['oracle_idxs_bert']

        source_sents = re.split(r'<s\d*>', source_annotated)
        source_sents = [x.strip() for x in source_sents if len(x.strip()) > 0]
        oracle = ' '.join([source_sents[i] for i in oracle_idx])

        oracle_rep = [oracle for _ in range(len(extracts))]
        _, _, f = bs.score(extracts, oracle_rep)
        f = f.cpu().numpy()

        cal_target_to_down.append(spearmanr(f, ea_rouge)[0])
        cal_to_down.append(spearmanr(calibration_scores, ea_rouge)[0])
        cal_to_target.append(spearmanr(f, calibration_scores))
        beam_to_down.append(spearmanr(list(range(len(f))), f)[0])

    print('Rank Corel between Calibration Target and Downstream ROUGE ', np.mean(cal_target_to_down))
    print('Rank Corel between Calibration Prediction and Downstream ROUGE ', np.mean(cal_to_down))
    print('Rank Corel between Extract Beam and Downstream ROUGE ', np.mean(beam_to_down))
    print('Rank Corel between Calibration Prediction and Calibration Target ', np.mean(cal_to_target))
