import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes:

    def __init__(self):
        self._ndim = None
        self._targets = None
        self._means = None
        self._vars = None
        self._target_prior = None

    def fit(self, x, y):
        self._ndim = x.shape[1]
        self._targets = np.sort(np.unique(y))
        self._means = np.zeros((len(self._targets), self._ndim))
        self._vars = np.zeros_like(self._means)
        self._target_prior = np.zeros(len(self._targets))
        for target in self._targets:
            self._means[target] = np.mean(x[y == target], axis=0)
            self._vars[target] = np.var(x[y == target], axis=0)
            self._target_prior[target] = (y == target).sum() / len(y)

    def predict(self, x):
        self._check_valid(x)
        posterior = np.zeros((len(x), len(self._targets)))
        for target_idx in self._targets:
            probs = np.ones(len(x))
            for feature_idx in range(self._ndim):
                probs *= self._gaussian_pdf(
                    self._means[target_idx, feature_idx],
                    self._vars[target_idx, feature_idx], x[:, feature_idx])
            probs *= self._target_prior[target_idx]
            # print(probs)
            posterior[:, target_idx] = probs
        return np.argmax(posterior, axis=1), posterior

    def test(self, x, y):
        self._check_valid(x, y)
        preds, posterior = self.predict(x)
        err = (preds != y).sum() / len(y)
        return preds, posterior, err

    def metric(self, x, y, pred, plot_confusion_matrix=True):
        self._check_valid(x, y)
        TP = ((pred == y)[y == 1]).sum()
        FN = ((pred != y)[y == 1]).sum()
        TN = ((pred == y)[y == 0]).sum()
        FP = ((pred != y)[y == 0]).sum()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        eps = 1e-7 if TP + FP == 0 else 0
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN)
        confusion_matrix = np.array([[TN, FN], [FP, TP]])
        print(f"{'Accuracy'.rjust(10)}: {accuracy:.7f}")
        print(f"{'Precision'.rjust(10)}: {precision:.7f}")
        print(f"{'Recall'.rjust(10)}: {recall:.7f}")
        if plot_confusion_matrix:
            self._plot_confusion_matrix(confusion_matrix)
            return accuracy, precision, recall, confusion_matrix
        return accuracy, precision, recall

    def plot_data(self, x, pred):
        plt.plot(x[pred == 0][:, 0], x[pred == 0][:, 1], 'xb')
        plt.plot(x[pred == 1][:, 0], x[pred == 1][:, 1], 'xr')
        plt.show()

    def roc_auc(self, y, y_score):
        tprs = []
        fprs = []
        counts = 0
        y_score = y_score.max(1)
        postive_score = y_score[y == 1]
        negtive_score = y_score[y == 0]
        thresholds = sorted(y_score, reverse=True)
        for thresh in thresholds:
            TP = ((y_score >= thresh)[y == 1]).sum()
            FN = ((y_score < thresh)[y == 1]).sum()
            TN = ((y_score < thresh)[y == 0]).sum()
            FP = ((y_score >= thresh)[y == 0]).sum()
            tprs.append(TP / (TP + FN))
            fprs.append(FP / (FP + TN))

        for pos_score in postive_score:
            for neg_score in negtive_score:
                if pos_score > neg_score:
                    counts += 1
                elif pos_score == neg_score:
                    counts += 0.5
                else:
                    pass
        auc = counts / (len(postive_score) * len(negtive_score))
        plt.plot(fprs, tprs)
        plt.xlabel("False Postive Rate")
        plt.ylabel("True Postive Rate")
        plt.show()
        print(f"AUC: {auc}")
        return auc

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_xticks(np.arange(len(self._targets)))
        ax.set_yticks(np.arange(len(self._targets)))
        ax.set_xticklabels(self._targets)
        ax.set_yticklabels(self._targets)

        thresh = (cm.min() + cm.max()) / 2
        for i in range(len(self._targets)):
            for j in range(len(self._targets)):
                color = 'w' if cm[i, j] < thresh else 'k'
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.tight_layout()
        plt.show()

    def _gaussian_pdf(self, mean, var, value):
        return np.exp(-(value - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def _check_valid(self, x, y=None):
        assert x.shape[1] == self._ndim
        if y is not None:
            assert np.isin(np.unique(y),
                           self._targets).sum() == len(np.unique(y))
