import os
import re

import numpy as np

with open('keywords.txt', 'r') as f:
    keywords = [aux.strip() for aux in f.readlines()]


def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email


def get_feature_vectors(body, from_mail, flag):
    if len(body) == 0:
        return None
    counters = np.zeros(len(keywords)+2)
    for i, word in enumerate(keywords):
        counters[i] = body.count(word)
    np.divide(counters, len(body))
    counters[-2] = len(from_mail)
    counters[-1] = flag
    return counters


if __name__ == "__main__":
    samples_array = []
    values_array = []
    for root, dirs, files in os.walk('./mails', topdown=False):
        for name in files:
            if 'bad' in name:
                phishing = 1
            else:
                phishing = 0

            with open(os.path.join(root, name), mode='r', errors='ignore') as f:
                mail_list = re.compile('\nFrom:').split(f.read())

            for i, x in enumerate(mail_list):
                raw_dict = parse_raw_message('from:' + x.lower())
                if raw_dict.get('body') is not None:
                    vector = get_feature_vectors(raw_dict['body'], raw_dict['from'], phishing)
                    if vector is not None:
                        samples_array.append(vector)
    samples = np.array(samples_array)
    np.random.shuffle(samples)
    np.save('input', samples)
