import os
from glob import glob
from summac.model_summac import SummaCConv
from datasets import load_from_disk
import argparse
import numpy as np
from tqdm import tqdm


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}



def infer_dataset(args, col):
    if args.dataset is None:
        rel_name = getattr(args, col)
        if 'samsum' in rel_name:
            args.dataset = 'samsum'
        elif 'nyt' in rel_name:
            args.dataset = 'nyt'
        elif 'cnn' in rel_name:
            args.dataset = 'cnn_dailymail'
        elif 'xsum' in rel_name:
            args.dataset = 'xsum'
        else:
            raise Exception(f'Cant infer dataset from {rel_name}. Must pass with --dataset flag.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add SummaC for Re-Ranked BRIO')

    parser.add_argument('--dataset', default=None)
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum/')
    parser.add_argument('--brio_exp', default='xsum_e_v1_ea')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--max_examples', default=9999999, type=int)

    args = parser.parse_args()

    infer_dataset(args, 'brio_exp')

    input_col, _ = summarization_name_mapping[args.dataset]

    pattern = os.path.expanduser(os.path.join('~', 'BRIO', 'result', args.brio_exp, 'candidate_ranking', '*.dec'))
    out_fn = os.path.expanduser(os.path.join('~', 'BRIO', 'result', args.brio_exp, 'summac.txt'))

    fns = list(glob(pattern))

    test = load_from_disk(os.path.join(args.data_dir, args.dataset + '_edu_alignments'))['test']
    assert len(test) == len(fns)
    sources = list(test[input_col])

    print(f'Computing Summac for {len(fns)} test set re-ranked examples')

    # https://github.com/tingofurro/summac/
    model_conv = SummaCConv(
        models=['vitc'], bins='percentile', granularity='sentence', nli_labels='e', device=args.device,
        start_file='default', agg='mean',
    )

    summaries = []
    ids = list(sorted([int(x.split('/')[-1].split('.')[0]) for x in fns]))
    for idx, id in tqdm(enumerate(ids), total=len(ids)):
        fn = os.path.join('/'.join(fns[0].split('/')[:-1]), f'{id}.dec')
        with open(fn, 'r') as fd:
            summary = fd.read().strip()
        summaries.append(summary)

    n = len(sources)
    if args.max_examples < n:
        print(f'Taking first {args.max_examples}/{n}')
        sources = sources[:args.max_examples]
        summaries = summaries[:args.max_examples]
    scores = model_conv.score(sources, summaries)['scores']
    print(len(scores))
    print(args.brio_exp)
    print(np.mean(scores))

    scores_str = list(map(str, scores))
    with open(out_fn, 'w') as fd:
        fd.writelines(scores_str)
