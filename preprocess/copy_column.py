from datasets import load_from_disk


if __name__ == '__main__':
    splits = ['test', 'validation', 'train']
    from_name = 'xsum'
    to_name = 'xsum_pegasus'

    to_dir = f'/nlp/projects/faithsum/{to_name}'
    out_dir = f'/nlp/projects/faithsum/{to_name}_w_bert'
    print(f'Run mv {out_dir} {to_dir}')

    from_data = load_from_disk(f'/nlp/projects/faithsum/{from_name}')
    to_data = load_from_disk(f'/nlp/projects/faithsum/{to_name}')

    for split in splits:
        id2oracle = dict(zip(from_data[split]['id'], from_data[split]['oracle_idxs_bert']))
        assert from_data[split]['summary'] == to_data[split]['summary']
        to_data[split] = to_data[split].map(lambda example: {'oracle_idxs_bert': id2oracle[example['id']]})

    print(to_data['test'].column_names)
    to_data.save_to_disk(out_dir)

    print(f'Run rm -rf {to_dir} && mv {out_dir} {to_dir}')
