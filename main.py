import numpy as np
from torchtext import data
import csv
import ipdb

# TODO: remove urls and perhaps substitute them with 'url'


def to_train_test_csv():
    tweets_full = 'data/SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
    tweets_test = 'data/twitter-2016test-A-English.txt'
    with open(tweets_full) as f:
        lines_full = tuple(
            line.split('\t')[:3] for line in f.read().split('\n')
            if len(line) > 0)
    with open(tweets_test) as f:
        lines_test = set(
            line.split('\t')[0] for line in f.read().split('\n')
            if len(line) > 0)

    def clean_tweet(tweet):
        tweet = tweet.replace("\\'", "'").replace('""', '"')
        return tweet if not (tweet[0] == tweet[-1]
                             and tweet[0] == '"') else tweet[1:-1]

    tweets_full = [(line[0], line[1], clean_tweet(line[2]))
                   for line in lines_full if len(line) == 3]

    # NB: file ids are identical
    with open('data/tweets_clean.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'tweet'])
        writer.writerows(tweets_full)

    print(len(tweets_full))


def main():
    fields = {
        'id':
        data.Field(),
        'label':
        data.Field(),
        'tweet':
        data.Field(lower=True,
                   tokenize='spacy',
                   batch_first=True,
                   tokenizer_language="en_core_web_sm")
    }
    train_dataset = data.TabularDataset('data/tweets_clean.csv',
                                        fields=fields.items(),
                                        format='csv',
                                        skip_header=True)
    for key, field in fields.items():
        field.build_vocab(train_dataset)

    ipdb.set_trace()

    data_iter = data.BucketIterator()


if __name__ == '__main__':
    # to_train_test_csv()
    main()
