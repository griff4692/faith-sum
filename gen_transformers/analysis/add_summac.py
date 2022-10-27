from summac.model_summac import SummaCConv


if __name__ == '__main__':
    # https://github.com/tingofurro/summac/
    model_conv = SummaCConv(
        models=['vitc'], bins='percentile', granularity="sentence", nli_labels='e', device='cuda', start_file='default',
        agg='mean'
    )

    document = "the sky is blue"
    summary = "the sky is blue"
    scores = model_conv.score([document], [summary])['scores']

    print(scores)
