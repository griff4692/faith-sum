import os
from glob import glob
from tqdm import tqdm
import ujson


from bert_score.scorer import BERTScorer


if __name__ == '__main__':
    batch_size = 256
    # dataset = 'xsum'
    # name = 'pegasus_xsum_top_beam'
    name = 'bart_large_cnn_top_beam'
    dataset = 'cnndm'

    path = os.path.expanduser(os.path.join('~', 'BRIO', dataset, name, 'diverse', 'test', '*.json'))
    print(path)

    bs = BERTScorer(lang='en')

    refs = []
    cands = []

    fns = list(glob(path))
    for fn in tqdm(fns):
        with open(fn, 'r') as fd:
            obj = ujson.load(fd)
            assert type(obj['candidates'][0][0]) == str
            cands.append(' '.join(obj['candidates'][0][0]))
            refs.append(obj['abstract'][0])

    _, _, f = bs.score(refs, cands, verbose=True, batch_size=batch_size)
    f = f.cpu().numpy()
    avg_bs = float(f.mean())
    print(avg_bs)
