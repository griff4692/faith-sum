import os
import itertools
import pandas as pd
import regex as re

BRIO_WEIGHTS = os.path.expanduser('~/BRIO/cache')


def get_rouge(line):
    scores = line.split(',')
    obj = {}
    for score in scores:
        val = float(score.split(':')[-1].strip())
        if 'rouge1' in score:
            obj['rouge1'] = val
        elif 'rouge2' in score:
            obj['rouge2'] = val
        else:
            obj['rougeLsum'] = val
    return obj


def extract_config(weight_dir):
    config_fn = os.path.join(BRIO_WEIGHTS, weight_dir, 'config.txt')
    with open(config_fn, 'r') as fd:
        config_blob = fd.read()
        lp = float(re.search(r'length_penalty=([^,;]+)', config_blob).group(1))
        scale = float(re.search(r'scale=([^,;]+)', config_blob).group(1))
        margin = float(re.search(r' margin=([^,;]+)', config_blob).group(1))
        mle_weight = float(re.search(r'mle_weight=([^,;]+)', config_blob).group(1))
        rank_weight = float(re.search(r'rank_weight=([^,;]+)', config_blob).group(1))
        lr = float(re.search(r'max_lr=([^,;]+)', config_blob).group(1))
        return {
            'scale': scale,
            'margin': margin,
            'mle_weight': mle_weight,
            'rank_weight': rank_weight,
            'length_penalty': lp,
            'learning_rate': lr,
        }


def extract_results(weight_dir, model_type, decode_method):
    config = extract_config(weight_dir)
    log_fn = os.path.join(BRIO_WEIGHTS, weight_dir, 'log.txt')
    rank_results = []
    gen_results = []
    best_r1_rank = 0
    best_r1_gen = 0
    with open(log_fn, 'r') as fd:
        lines = list(map(lambda x: x.strip(), fd.readlines()))
        lines = [x for x in lines if 'rouge' in x]

        for line in lines:
            r = get_rouge(line)
            if 'ranking' in line:
                best_r1_rank = max(r['rouge1'], best_r1_rank)
                rank_results.append(r)
            else:
                best_r1_gen = max(r['rouge1'], best_r1_gen)
                gen_results.append(r)

    outputs = []
    for step in range(len(rank_results)):
        rr = rank_results[step]
        rr['step'] = step
        rr['eval_type'] = 'rank'
        rr['model_type'] = model_type
        rr['experiment'] = weight_dir
        rr['best_rouge1'] = best_r1_rank
        rr['decode_method'] = decode_method
        rr.update(config)

        gr = gen_results[step]
        gr['step'] = step
        gr['eval_type'] = 'generation'
        gr['model_type'] = model_type
        gr['experiment'] = weight_dir
        gr['best_rouge1'] = best_r1_gen
        gr['decode_method'] = decode_method
        gr.update(config)
        outputs.append(rr)
        outputs.append(gr)
    return outputs


if __name__ == '__main__':
    overwrite = False
    initial_tune = False

    if initial_tune:
        processed_fn = os.path.expanduser('~/brio_samsum_results.csv')
        if not os.path.exists(processed_fn) or overwrite:
            weights = os.listdir(BRIO_WEIGHTS)
            bl_ctx = [d for d in weights if 'bart_large_samsum_ctx' in d]
            bl_mul = [d for d in weights if 'bart_large_samsum_mul' in d]

            ea_ctx = [d for d in weights if 'samsum_extract_generator_ctx' in d]
            ea_mul = [d for d in weights if 'samsum_extract_generator_mul' in d]

            bl_ctx_results = list(itertools.chain(*[extract_results(
                x, model_type='ctx', decode_method='beam_search') for x in bl_ctx]))
            bl_mul_results = list(itertools.chain(*[extract_results(
                x, model_type='mul', decode_method='beam_search') for x in bl_mul]))

            ea_ctx_results = list(itertools.chain(*[extract_results(
                x, model_type='ctx', decode_method='extract_abstract') for x in ea_ctx]))
            ea_mul_results = list(itertools.chain(*[extract_results(
                x, model_type='mul', decode_method='extract_abstract') for x in ea_mul]))

            all_results = pd.DataFrame(bl_ctx_results + bl_mul_results + ea_ctx_results + ea_mul_results)

            print(f'Saving results to {processed_fn}')
            all_results.to_csv(processed_fn, index=False)

        else:
            all_results = pd.read_csv(processed_fn)

        all_results['is_best'] = all_results['best_rouge1'] == all_results['rouge1']

        ctx_results = all_results[all_results['eval_type'] == 'rank']
        ctx_results = ctx_results[ctx_results['is_best']]
        ctx_results = ctx_results.sort_values(by=['decode_method', 'experiment'])
        excel_str = []
        col_order = ['decode_method', 'experiment', 'scale', 'margin', 'length_penalty', 'mle_weight', 'rank_weight', 'learning_rate', 'rouge1', 'rouge2', 'rougeLsum']
        for record in ctx_results.to_dict('records'):
            excel_str.append(' '.join([str(record[c]) for c in col_order]))

        print('\n'.join(excel_str))

        mul_results = all_results[(all_results['model_type'] == 'mul') & (all_results['eval_type'] == 'generation')]
        mul_results = mul_results[mul_results['is_best']]
        mul_results = mul_results.sort_values(by=['decode_method', 'experiment'])
        excel_str = []
        col_order = ['decode_method', 'experiment', 'scale', 'margin', 'length_penalty', 'mle_weight', 'rank_weight', 'learning_rate', 'rouge1', 'rouge2', 'rougeLsum']
        for record in mul_results.to_dict('records'):
            excel_str.append(' '.join([str(record[c]) for c in col_order]))

        print('Generation Results...')
        print('\n'.join(excel_str))
    else:
        weights = os.listdir(BRIO_WEIGHTS)
        samsum_rerank = [d for d in weights if 'samsum_rerank' in d]

        results = list(itertools.chain(*[extract_results(
            x, model_type='rank', decode_method='extract_abstract') for x in samsum_rerank]))

        all_results = pd.DataFrame(results)
        all_results['is_best'] = all_results['best_rouge1'] == all_results['rouge1']
        processed_fn = os.path.expanduser('~/brio_samsum_rerank_results.csv')
        print(f'Saving results to {processed_fn}')
        all_results.to_csv(processed_fn, index=False)

        ctx_results = all_results[all_results['eval_type'] == 'rank']
        ctx_results = ctx_results[ctx_results['is_best']]
        ctx_results = ctx_results.sort_values(by=['experiment'])
        excel_str = []
        col_order = ['decode_method', 'experiment', 'scale', 'margin', 'length_penalty', 'mle_weight', 'rank_weight', 'learning_rate', 'rouge1', 'rouge2', 'rougeLsum']
        for record in ctx_results.to_dict('records'):
            excel_str.append(' '.join([str(record[c]) for c in col_order]))

        print('\n'.join(excel_str))