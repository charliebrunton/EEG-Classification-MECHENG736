import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeavePOut

def cross_validate_linear(features, labels):
    # perform leave-2-out cross-validation using LDA (Fishers linear discriminant)
    # inputs: features -> numpy array of shape (n_samples, n_features) ie. 36 epochs (rows), each with alpha/theta power features, 
    # labels -> associated labels (0 = eyes open)
    # output: average classification accuracy

    lda = LinearDiscriminantAnalysis() # creates LDA classifier using class in scikit-learn library
    lpo = LeavePOut(p=2) # hold out 2 epochs each 'fold'
    accuracies = []

    for train_idxs, test_idxs in lpo.split(features): # lpo.split(features) -> returns a generator that yields (training epochs, testing epochs) pairs for each fold,
        # ie. for every combination that holds 2 epochs

        lda.fit(features[train_idxs], labels[train_idxs]) # train classifier on 34 epochs
        accuracy = lda.score(features[test_idxs], labels[test_idxs]) # test on 2 held out epochs
        accuracies.append(accuracy)
        
    return np.mean(accuracies)
