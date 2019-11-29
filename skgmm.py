from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import math

class GMMSet:

    def __init__(self, gmm_order = 1):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)

    def gmm_score(self, gmm, x):
        # print(gmm.score(x))
        return np.sum(gmm.score(x))

    @staticmethod
    def softmax(scores):
        scores_sum = sum([math.exp(i) for i in scores])
        score_max  = math.exp(max(scores))
        return round(score_max / scores_sum, 3)

    def predict_one(self, x):
        # print(x.shape)
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]
        p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)
        # print("-----------------------")
        # print(p)
        # print(scores)
        p = [(str(self.y[i]), y, p[0][1] - y) for i, y in p]
        # print(p)
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]
        # print(result)
        p = max(result, key=operator.itemgetter(1))
        # print(p)
        softmax_score = self.softmax(scores)
        return p[0], softmax_score

    def before_pickle(self):
        pass

    def after_pickle(self):
        pass
