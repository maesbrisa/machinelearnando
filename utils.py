import numpy as np

with open('./keywords.txt', 'r') as f:
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


def word_counter(text, counters):
    for i, word in enumerate(keywords):
        counters[i] = text.count(word)
    np.divide(counters, len(text))
    return counters
