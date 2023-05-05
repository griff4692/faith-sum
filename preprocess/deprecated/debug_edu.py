from collections import Counter
import regex as re
import ujson
import os
from glob import glob

from p_tqdm import p_uimap


def is_valid(html):
    html_no_space = re.sub(r'\s+', '', html)
    assert len(re.findall('</e>', html_no_space)) == len(re.findall('<e>', html_no_space))
    assert html_no_space.startswith('<e>') and html_no_space.endswith('</e>')

    # Assert that when we close a tag we immediately open another. Except for final closure
    return len(re.findall('</e>', html_no_space)) == len(re.findall('</e><e>', html_no_space)) + 1


def validate_example(fn):
    if fn.endswith('jsonl'):
        with open(fn, 'r') as fd:
            lines = [x for x in fd.readlines() if len(x.strip()) > 0]
            for line in lines:
                obj = ujson.loads(line)

                sa = obj['source_edu_annotated']
                ta = obj['target_edu_annotated']

                if is_valid(sa) and is_valid(ta):
                    return None
                return fn
    else:
        assert fn.endswith('json')
        with open(fn, 'r') as fd:
            obj = ujson.load(fd)

            sents_w_edu = [x['sent_w_edu'] for x in obj]
            for s in sents_w_edu:
                if not is_valid(s):
                    return fn
        return None


def get_error_cts(fn):
    assert fn.endswith('json')
    with open(fn, 'r') as fd:
        obj = ujson.load(fd)
        return any(['error' in x for x in obj])


if __name__ == '__main__':
    alex_pattern = True
    check_for_errors = True
    if alex_pattern:
        pattern = '/nlp/projects/faithsum/edu/*/*/*.json'
    else:
        pattern = os.path.expanduser(os.path.join('~', 'edu_alignments_sample', '*'))

    func = get_error_cts if check_for_errors else validate_example
    fns = list(glob(pattern))
    print(f'{len(fns)} files matching pattern')
    invalid_fns = list(filter(None, list(p_uimap(func, fns))))

    print(f'{len(invalid_fns)} invalid fns')
    counts = []
    for fn in invalid_fns:
        if 'cnn' in fn:
            counts.append('cnn')
        elif 'nyt' in fn:
            counts.append('nyt')
        elif 'xsum' in fn:
            counts.append('xsum')

    print(Counter(counts).most_common())
