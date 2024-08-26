#Task C: implement Knn

from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial import distance

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # TODO: complete
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X):
        
        # Note: You can use self.n_neighbors here
        # K = n_neighbors, m = number of test samples, d = number of features

        distances = distance.cdist(X, self.X_train) # time complexity O(m * d)
        minDictancesLabels = self.y_train.to_numpy()[distances.argpartition(self.n_neighbors)] # time complexity O(m)
        predictions = np.sign(np.sum(minDictancesLabels[:, :self.n_neighbors], axis=1)) # time complexity O(m)
        return np.asarray(predictions)
