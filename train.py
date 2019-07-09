import numpy as np
import time
from sklearn import svm

def split_vectors_results(arr):
    return np.hsplit(arr, [arr.shape[1]-1])
    

if __name__ == "__main__":
    path = './input.npy'
    array = np.load(path)
    print('cargao')
    [samples, results] = split_vectors_results(array)
    [sample_train,validation_train] = np.vsplit(samples, [6000])
    [sample_result, validation_result] =  np.vsplit(results, [6000])
    intento = svm.SVC(kernel='linear')
    intento.fit(sample_train, sample_result)
    print('trained')
    errors = []
    for y, m in enumerate(validation_train):
        test = intento.predict(m.reshape(1,-1))[0]
        errors.append(abs(test - validation_result[y][0]))
    print(np.average(errors))

