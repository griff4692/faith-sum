MLE_COEF = [0.0, 0.1, 1, 10.0]
PLAN_COEF = [0.1, 1, 10.0]


if __name__ == '__main__':
    for dataset, data_name, extra_args in [
        ['cnn_dailymail', 'cnn', ''],
        # ['nyt', 'nyt', ''],
        ['xsum', 'xsum', '--lr 1e-4']
    ]:
        for mle in MLE_COEF:
            outputs = []
            for plan in PLAN_COEF:
                if mle == plan == 0:
                    continue
                like = unlike = plan
                experiment = f'{data_name}_ea_final_mle_{mle}_like_{like}_unlike_{unlike}'
                cmd = f'python main.py --per_device_train_bs 2 --summary_style abstract -extract_indicators ' \
                      f'--dataset {dataset} --mle_weight {mle} --like_coef {like} --unlike_coef {unlike} ' \
                      f'--experiment {experiment} {extra_args}'.strip()
                row = [experiment, data_name, str(mle), str(like), str(unlike)]
                row_str = ' '.join(row)
                # print(row_str)
                print(cmd)
