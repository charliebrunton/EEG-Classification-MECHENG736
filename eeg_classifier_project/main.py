import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from scripts.power_features import compute_band_powers
from scripts.linear_classifier import cross_validate_linear
from scripts.svm_classifier import cross_validate_svm
from scripts.plot_powers import plot_band_powers
from scripts.plot_decision_boundary import plot_decision_boundary
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeavePOut
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar

# ensure plots folder exists
os.makedirs("plots", exist_ok = True)

# import and segment data
fs, data = wav.read("data/recording2.wav")
# print(fs)
# print(data.shape)
epoch_length = 5 * fs # 50000 samples per epoch
data = data[:epoch_length * 36] # trim extra samples
epochs = [data[i:i+epoch_length] for i in range(0, len(data), epoch_length)]
labels = np.array([1 if i % 2 == 0 else 0 for i in range(0, 36)])

# extract features
features = np.array([compute_band_powers(epoch, fs) for epoch in epochs])
alpha_powers = features[:, 0]
theta_powers = features[:, 1]
plot_band_powers(alpha_powers, theta_powers, labels)

# linear classifier
lda = LinearDiscriminantAnalysis()
lda.fit(features, labels)
linear_acc = cross_validate_linear(features, labels)
print(f"Linear classifier accuracy: {linear_acc * 100:.2f}%")
plot_decision_boundary(lda, features, labels, "Linear classifier decision boundary")

# svm classifier
svm = SVC(kernel = 'linear')
svm.fit(features, labels)
svm_acc = cross_validate_svm(features, labels)
print(f"SVM classifier accuracy: {svm_acc * 100:.2f}%")
plot_decision_boundary(svm, features, labels, "SVM classifier decision boundary")

# for Q2
lda = LinearDiscriminantAnalysis()
lpo = LeavePOut(p=2)

for train_idxs, test_idxs in lpo.split(features):
    lda.fit(features[train_idxs], labels[train_idxs])
    plot_decision_boundary(lda, features, labels, "q2", test_idxs=test_idxs)
    break # only take first fold for plotting

# for Q4
# load file
fs, data = wav.read("data/ME736_EEGLab_Q4b.wav")
t = np.arange(len(data)) / fs # time axis [s]

# plot
plt.figure(figsize=(10, 4))
plt.plot(t, data, color='k', linewidth=0.8)
plt.axvline(5, color='r', linestyle='--', label='Eyes open starts')

plt.xlabel("Time [s]")
plt.ylabel("EEG amplitude")
plt.legend()

# save plot
file_name = "plots/q4.png"
plt.savefig(file_name, dpi = 300)

# for Q5
linear_img = mpimg.imread("plots/linear_classifier_decision_boundary.png")
svm_img = mpimg.imread("plots/svm_classifier_decision_boundary.png")

# plot together
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(linear_img)
axes[0].axis('off')
axes[1].imshow(svm_img)
axes[1].axis('off')
plt.tight_layout()

# save plot
file_name = "plots/q5.png"
plt.savefig(file_name, dpi = 300)

# check for statistically significant difference -> McNemars test
# code taken from ChatGPT:
def compare_classifiers(features, labels):
    lda = LinearDiscriminantAnalysis()
    svm = SVC(kernel='linear')
    lpo = LeavePOut(p=2)

    # track disagreements
    n01, n10 = 0, 0

    for train_idx, test_idx in lpo.split(features):
        # fit both classifiers
        lda.fit(features[train_idx], labels[train_idx])
        svm.fit(features[train_idx], labels[train_idx])

        # predict on held-out epochs
        lda_pred = lda.predict(features[test_idx])
        svm_pred = svm.predict(features[test_idx])

        # compare to truth
        truth = labels[test_idx]
        lda_correct = (lda_pred == truth)
        svm_correct = (svm_pred == truth)

        # update contingency counts
        for l_ok, s_ok in zip(lda_correct, svm_correct):
            if not l_ok and s_ok:
                n01 += 1   # FLD wrong, SVM right
            elif l_ok and not s_ok:
                n10 += 1   # FLD right, SVM wrong

    # build 2x2 table
    # only disagreements matter for McNemar’s test -> correct/correct and wrong/wrong pairs not counted
    table = np.array([[0, n01],
                      [n10, 0]])

    # run McNemar’s test
    result = mcnemar(table, exact=True)
    print("Contingency table:")
    print(table)
    print(f"McNemar statistic = {result.statistic}, p = {result.pvalue}")

compare_classifiers(features, labels)
