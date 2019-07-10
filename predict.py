import numpy as np
from sklearn.externals import joblib

from utils import parse_raw_message, word_counter, keywords


def get_feature_vectors(body, from_mail):
    if len(body) == 0:
        return None
    counters = np.zeros(len(keywords)+1)
    counters = word_counter(body, counters)
    counters[-1] = len(from_mail)
    return counters


def predict_phishing(body):
    raw_dict = parse_raw_message(body)
    if raw_dict.get('body') is not None:
        vector = get_feature_vectors(raw_dict['body'], raw_dict['from'])
        if vector is not None:
            svm_trained = joblib.load('svm.joblib')
            return svm_trained.predict_proba(vector.reshape(1, -1))
