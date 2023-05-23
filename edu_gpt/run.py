import os
from glob import glob

import argparse
import numpy as np
import openai
from tqdm import tqdm
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION


openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=3)
def chat_gpt(messages, model='gpt-4', temperature=0.3, max_tokens=256):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return response['choices'][0].message.content


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def instruct_gpt(prompt, model='gpt-4', temperature=0.3, max_tokens=256):
    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )
    return response['choices'][0].text


def get_dataset_idx(x):
    return x.split('/')[-1].replace('.txt', '').split('_')[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-3.5 / GPT-4.5')

    # Configuration Parameters
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--extract_experiment', default='cnn_e_final')
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--max_examples', default=16, type=int)
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--mode', default='pga', choices=['vanilla', 'pga'])

    args = parser.parse_args()

    in_dir = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, args.mode + '_prompts')
    assert os.path.exists(in_dir)
    out_dir = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, args.model)
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    is_chat = args.model in {'gpt-3.5-turbo', 'gpt-4'}

    fns = list(glob(os.path.join(in_dir, '*.txt')))

    ids = list(sorted(list(set([get_dataset_idx(x) for x in fns]))))
    if args.max_examples is not None and len(ids) > args.max_examples:
        np.random.seed(1992)
        np.random.shuffle(ids)
        ids = ids[:args.max_examples]
    ids = set(ids)
    fns_filt = [fn for fn in fns if get_dataset_idx(fn) in ids]

    for fn in tqdm(fns_filt):
        out_fn = os.path.join(out_dir, fn.split('/')[-1])
        if os.path.exists(out_fn) and not args.overwrite:
            print('Skipping ', out_fn)
            continue
        with open(fn, 'r') as fd:
            prompt = fd.read()

            if is_chat:
                messages = [
                    # Boost its ego first
                    {'role': 'system', 'content': 'You are a helpful and concise assistant for text summarization.'},
                    {'role': 'user', 'content': prompt}
                ]
                output = chat_gpt(messages, model=args.model)
            else:
                output = instruct_gpt(prompt=prompt, model=args.model)

            with open(out_fn, 'w') as fd:
                fd.write(output)
