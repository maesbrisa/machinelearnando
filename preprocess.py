import numpy as np


with open('palabras_clave_low.txt', 'r') as f:
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


def get_feature_vectors(body):
    counters = np.zeros(len(keywords))
    for i, word in enumerate(keywords):
        counters[i] = body.count(word)
    return np.divide(counters, len(body)), 1


if __name__ == "__main__":
    path = './bad.txt'
    with open(path, mode='r', errors='ignore') as f:
        mail_list = f.read().split('From')

    samples_array = []
    values_array = []

    for i, x in enumerate(mail_list):
        raw_message = parse_raw_message('from:' + x.lower()).get('body')
        if raw_message is not None:
            vector, result = get_feature_vectors(raw_message)
            samples_array.append(vector)
            values_array.append(result)

    print(len(values_array))
    with open('./vectors.txt', 'w') as f:
        print(np.array(samples_array), file=f)
    with open('./results.txt', 'w') as f:
        print(np.array(values_array), file=f)
