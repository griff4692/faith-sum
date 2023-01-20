from datasets import load_from_disk


if __name__ == '__main__':
    dataset = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')['validation']

    print('here')  # 61.7, 62.7