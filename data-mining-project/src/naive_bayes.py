import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / n_samples
            self.feature_likelihoods[cls] = [
                (np.mean(X_cls[:, feature]), np.var(X_cls[:, feature]) + 1e-9) for feature in range(n_features)
            ]

    def _calculate_likelihood(self, x, cls):
        likelihood = 1.0
        for feature_index in range(len(x)):
            mean, var = self.feature_likelihoods[cls][feature_index]
            # Tránh chia cho 0
            if var == 0:
                var = 1e-9
            likelihood *= (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x[feature_index] - mean) ** 2) / (2 * var))
        return likelihood

    def predict(self, X):
        predictions = []
        for x in X:
            class_probabilities = {}
            for cls in self.classes:
                likelihood = self._calculate_likelihood(x, cls)
                class_probabilities[cls] = likelihood * self.class_priors[cls]
            predictions.append(max(class_probabilities, key=class_probabilities.get))
        return np.array(predictions)

    def score(self, X, y):
        """
        Trả về độ chính xác dự đoán trên tập dữ liệu X, y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)