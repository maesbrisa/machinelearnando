import numpy as np
from sklearn import svm
from joblib import dump
    

if __name__ == "__main__":
    path = './input.npy'
    array = np.load(path)
    [samples, results] = np.hsplit(array, [array.shape[1]-1])

    [sample_train, validation_train] = np.vsplit(samples, [6000])
    [sample_result, validation_result] = np.vsplit(results, [6000])

    vector_machine = svm.SVC(kernel='linear')
    vector_machine.fit(sample_train, sample_result.ravel())

    errors = []
    for y, m in enumerate(validation_train):
        test = vector_machine.predict(m.reshape(1, -1))[0]
        errors.append(abs(test - validation_result[y][0]))
    print(np.average(errors))
    dump(vector_machine, 'svm.joblib')
