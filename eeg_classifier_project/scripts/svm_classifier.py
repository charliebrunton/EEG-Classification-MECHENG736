import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeavePOut

def cross_validate_svm(features, labels):
    # perform leave-2-out cross-validation using SVM (support vector machine)
    # inputs: features -> numpy array of shape (n_samples, n_features) ie. 36 epochs (rows), each with alpha/theta power features, 
    # labels -> associated labels (0 = eyes open)
    # output: average classification accuracy

    svm = SVC(kernel = 'linear') # create linear SVM classifier
    lpo = LeavePOut(p = 2) # hold out 2 epochs for each fold
    accuracies = []

    # same process as in linear_classifier.py
    for train_idx, test_idx in lpo.split(features):
        svm.fit(features[train_idx], labels[train_idx])
        accuracy = svm.score(features[test_idx], labels[test_idx])
        accuracies.append(accuracy)

    return np.mean(accuracies)
