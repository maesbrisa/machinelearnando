import numpy as np
import os

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


def get_feature_vectors(from_mail, body):
    if len(body) == 0:
        return None, None
    counters = np.zeros(len(keywords)+1)
    for i, word in enumerate(keywords):
        counters[i] = body.count(word)
    counters[-1] = len(from_mail)
    return np.divide(counters, len(body))


if __name__ == "__main__":
    samples_array = []
    values_array = []
    for root, dirs, files in os.walk('./mails', topdown=False):
        for name in files:
            if 'bad' in name:
                phishing = True
            else:
                phishing = False

            with open(os.path.join(root, name), mode='r', errors='ignore') as f:
                mail_list = f.read().split('From')

            for i, x in enumerate(mail_list):
                raw_dict = parse_raw_message('from:' + x.lower())
                if raw_dict.get('body') is not None:
                    vector = get_feature_vectors(raw_dict['from'], raw_dict['body'])
                    if vector is not None:
                        samples_array.append(vector)
                        values_array.append(int(phishing))

    with open('./vectors.txt', 'w') as f:
        print(np.array(samples_array), file=f)
    with open('./results.txt', 'w') as f:
        print(np.array(values_array), file=f)
