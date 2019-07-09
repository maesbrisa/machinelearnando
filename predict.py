import os
import re
from utils import parse_raw_message, word_counter
import argparse
from sklearn.externals import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help='File to check')
args = parser.parse_args()

with open('keywords.txt', 'r') as f:
    keywords = [aux.strip() for aux in f.readlines()]


def get_feature_vectors(body, from_mail):
    if len(body) == 0:
        return None
    counters = np.zeros(len(keywords)+1)
    counters = word_counter(body, counters)
    counters[-1] = len(from_mail)
    return counters


if __name__ == "__main__":
    samples_array = []

    with open(args.file, mode='r', errors='ignore') as f:
        mail_list = re.compile('\nFrom:').split(f.read())
    for i, x in enumerate(mail_list):
                raw_dict = parse_raw_message('from:' + x.lower())
                if raw_dict.get('body') is not None:
                    vector = get_feature_vectors(raw_dict['body'], raw_dict['from'])
                    if vector is not None:
                        samples_array.append(vector)
    samples = np.array(samples_array)
    # np.save('input', samples)
    svm_trained = joblib.load('svm.joblib')
    output = []
    for s in samples:
        r = svm_trained.predict(s.reshape(1, -1))[0]
        output.append(r)
        print(r)
