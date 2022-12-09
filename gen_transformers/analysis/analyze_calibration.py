import pandas as pd
import regex as re

from datasets import load_from_disk
import numpy as np

from bert_score.scorer import BERTScorer
from tqdm import tqdm

from scipy.stats import spearmanr, pearsonr
import scipy.stats as ss


def remove_na(arr):
    return [x for x in arr if not np.isnan(x)]


if __name__ == '__main__':
    # experiment = 'samsum_brio_score_bert_extract'
    experiment = 'bart_large_samsum'
    split = 'test'
    cal_col = 'calibrated_beam_score'
    fn = f'{split}_from_beam_16_extract'
    fn = 'test_beam_16_outputs'
    # fn = 'test_from_beam_16_extract_w_unprompted'
    add_bs = False
    df = pd.read_csv(f'/nlp/projects/faithsum/results/{experiment}/{fn}.csv')
    dataset = None
    if dataset is None:
        if 'cnn' in experiment:
            dataset = 'cnn_dailymail'
        elif 'samsum' in experiment:
            dataset = 'samsum'
        else:
            dataset = 'xsum'

    bs = None
    if add_bs:
        bs = BERTScorer(model_type='microsoft/deberta-large-mnli', batch_size=4)
        dataset = load_from_disk(f'/nlp/projects/faithsum/{dataset}')[split]

    beam_pearson = []
    pearson = []
    corels = []
    beam_to_down = []
    cal_target_to_down = []
    cal_to_down = []
    cal_to_extract = []
    beam_to_extract = []
    cal_to_target = []
    ensemble = []
    records = df.to_dict('records')

    if 'from_extract_rouges' in records[0]:
        for record in tqdm(records):
            e_rouge = [float(x) for x in record['extract_rouges'].split(',')]
            ea_rouge = [float(x) for x in record['from_extract_rouges'].split('<cand>')]
            beam_to_extract.append(spearmanr(list(range(len(e_rouge))), e_rouge)[0])
            beam_to_down.append(spearmanr(list(range(len(e_rouge))), ea_rouge)[0])
            beam_pearson.append(pearsonr(list(range(len(e_rouge))), ea_rouge)[0])
            f = None
            if add_bs:
                dataset_idx = record['dataset_idx']
                source_annotated = dataset[dataset_idx]['source_annotated']
                oracle_idx = dataset[dataset_idx]['oracle_idxs_bert']

                source_sents = re.split(r'<s\d*>', source_annotated)
                source_sents = [x.strip() for x in source_sents if len(x.strip()) > 0]
                oracle = ' '.join([source_sents[i] for i in oracle_idx])
                extracts = record['extract'].split('<cand>')
                oracle_rep = [oracle for _ in range(len(extracts))]
                _, _, f = bs.score(extracts, oracle_rep)
                f = f.cpu().numpy()

                cal_target_to_down.append(spearmanr(f, ea_rouge)[0])

            if cal_col in record:
                calibration_scores = [float(x) for x in record[cal_col].split(',')]
                cal_to_extract.append(spearmanr(calibration_scores, e_rouge)[0])
                if f is not None:
                    cal_to_target.append(spearmanr(f, calibration_scores))
                cal_to_down.append(spearmanr(calibration_scores, ea_rouge)[0])
                pearson.append(pearsonr(calibration_scores, ea_rouge)[0])

                cal_rank = ss.rankdata(-np.array(calibration_scores)) - 1
                beam_rank = np.arange(len(calibration_scores))
                ensemble_rank = [-(a + b) / 2.0 for a, b in zip(cal_rank, beam_rank)]
                ensemble.append(spearmanr(ensemble_rank, ea_rouge)[0])
    else:
        p, s = [], []
        for record in tqdm(records):
            a_rouge = [float(x) for x in record['abstract_rouges'].split(',')]
            p.append(pearsonr(a_rouge, np.arange(len(a_rouge)))[0])
            s.append(spearmanr(a_rouge, np.arange(len(a_rouge)))[0])

        print('Rank Corel between Abstract Beam and Downstream ROUGE ', np.mean(remove_na(s)))
        print('Pearson Corel between Abstract Beam and Downstream ROUGE ', np.mean(remove_na(p)))

        exit(0)

    beam_to_down = remove_na(beam_to_down)
    beam_to_extract = remove_na(beam_to_extract)
    beam_pearson = remove_na(beam_pearson)
    cal_to_extract = remove_na(cal_to_extract)
    cal_to_down = remove_na(cal_to_down)
    ensemble = remove_na(ensemble)
    pearson = remove_na(pearson)

    print('Rank Corel between Extract Beam and Downstream ROUGE ', np.mean(beam_to_down))
    print('Rank Corel between Extract Beam and Extract ROUGE ', np.mean(beam_to_extract))
    print('Pearson Corel between Extract Beam and Downstream ROUGE ', np.mean(beam_pearson))

    if len(cal_to_down) > 0:
        print('Rank Corel between Calibration Prediction and Extract ROUGE ', np.mean(cal_to_extract))
        print('Rank Corel between Calibration Prediction and Downstream ROUGE ', np.mean(cal_to_down))
        print('Rank Corel between Ensemble Calibration Score and Downstream ROUGE ', np.mean(ensemble))
        print('Pearson Corel between Calibration Prediction and Downstream ROUGE ', np.mean(pearson))

    if len(cal_to_target) > 0:
        print('Rank Corel between Calibration Prediction and Calibration Target ', np.mean(cal_to_target))
        print('Rank Corel between Calibration Target and Downstream ROUGE ', np.mean(cal_target_to_down))
