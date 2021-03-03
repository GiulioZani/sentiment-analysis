import spacy
import numpy as np
import json


def main():
    file_name = "SemEval2017-task4-dev.subtask-A.english.INPUT.txt"
    with open(file_name) as f:
        raw = f.read()
    data = [line.split('\t')[:-1] for line in raw.split('\n')]
    data = [
        tuple(l.replace('\\', '') for l in d) for d in data
        if len(d) == 3 and min([len(el) for el in d]) != 0
    ]
    text = ' '.join(tuple(c[-1].lower() for c in data))
    vocab = text.split()
    print(len(text))
    # nlp = spacy.blank("en")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = len(text)
    doc = nlp(text)
    vocab = list(doc.vocab.strings)
    print(len(vocab))
    # print(data[np.random.randint(0, 500)])


if __name__ == '__main__':
    main()
