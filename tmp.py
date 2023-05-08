MLE_COEF = [0.1, 0.5, 1]
LIKE_COEF = [0.1, 0.5, 1]
UNLIKE_COEF = [0.1, 0.5, 1]


if __name__ == '__main__':
    for dataset, data_name in [
        ['cnn_dailymail', 'cnn'],
        ['nyt', 'nyt'],
        ['xsum', 'xsum']
    ]:
        for mle in MLE_COEF:
            for like in LIKE_COEF:
                for unlike in UNLIKE_COEF:
                    outputs = []
                    experiment = f'{data_name}_ea_final_mle_{mle}_like_{like}_unlike_{unlike}'
                    cmd = f'python main.py --per_device_train_bs 2 --summary_style abstract -extract_indicators ' \
                          f'--dataset {dataset} --mle_weight {mle} --like_coef {like} --unlike {unlike} ' \
                          f'--experiment {experiment}'
                    row = [experiment, data_name, str(mle), str(like), str(unlike)]  #, cmd]
                    row_str = ' '.join(row)
                    # print(row_str)
                    print(cmd)
