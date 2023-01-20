import os
import ujson
from tqdm import tqdm


if __name__ == '__main__':
    dataset = 'xsum'
    brio_dir = os.path.expanduser(os.path.join('~', 'BRIO', dataset))
    cand_dir = os.path.join(brio_dir, 'xsum_e_v1_ea', 'diverse', 'test')
    add_dir = os.path.join(brio_dir, 'xsum_ea_reg_1.0_0.1_0.1_unprompted', 'diverse', 'test')
    out_dir = os.path.join(brio_dir, f'{dataset}_combined', 'diverse', 'test')

    print(f'Saving to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    n = len(os.listdir(cand_dir))

    for idx in tqdm(range(n)):
        with open(os.path.join(add_dir, f'{idx}.json'), 'r') as fd:
            to_add = ujson.load(fd)

        with open(os.path.join(cand_dir, f'{idx}.json'), 'r') as fd:
            obj = ujson.load(fd)

        assert obj['article'] == to_add['article']

        assert len(to_add['candidates_untok']) == 1
        obj['candidates_untok'][-1] = to_add['candidates_untok'][0]
        assert len(to_add['candidates']) == 1
        obj['candidates'][-1] = to_add['candidates'][0]

        with open(os.path.join(out_dir, f'{idx}.json'), 'w') as fd:
            ujson.dump(obj, fd)
