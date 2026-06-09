import numpy as np


class FisherBinaryClassifier:
    """A simple Fisher's linear discriminant binary classifier.

    Attributes:
        class_means: A dictionary mapping class labels to their mean vectors.
        overall_mean: The overall mean vector of the data.
        scatter_within: The within-class scatter matrix.
        scatter_between: The between-class scatter matrix.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X_train, y_train):
        """Fit the Fisher's linear discriminant model to the training data.

        Args:
            X_train (list[list[float]]): Training feature vectors.
            y_train (list): Training labels corresponding to each row in X_train.
        """
        # Compute class means, overall mean, and scatter matrices here
        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]
        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        S0 = np.zeros((X_train.shape[1], X_train.shape[1]))
        for x in X0:
            d = (x - mu0).reshape(-1, 1)
            S0 += d @ d.T

        S1 = np.zeros((X_train.shape[1], X_train.shape[1]))
        for x in X1:
            d = (x - mu1).reshape(-1, 1)
            S1 += d @ d.T

        SW = S0 + S1

        self.w = np.linalg.pinv(SW) @ (mu1 - mu0)

        m0 = self.w @ mu0
        m1 = self.w @ mu1

        self.threshold = (m0 + m1) / 2
        self._fitted = True

    def predict(self, X_test):
        """Predict labels for a collection of test examples.

        Args:
            X_test (list[list[float]]): Test examples to classify.
        Returns:
            list: Predicted labels for each test example.
        """
        if not self._fitted:
            raise ValueError("The model has not been fitted yet.")

        scores = self.decision_function(X_test)

        return (
            scores >= 0
        ).astype(bool)


class FisherClassifier:
    """A simple OvR (One vs Rest) Fisher's linear discriminant classifier.
    This class uses multiple binary classifiers to handle multi-class classification problems.
    """

    def __init__(self):
        self._num_classes = 0
        self._binary_classifiers = []
        self._unique_classes = []
        self._fitted = False

    def fit(self, X_train, y_train):
        """Fit the Fisher's linear discriminant model to the training data.

        Args:
            X_train (list[list[float]]): Training feature vectors.
            y_train (list): Training labels corresponding to each row in X_train.
        """
        # Compute class means, overall mean, and scatter matrices here
        unique_classes = np.unique(y_train)
        self._num_classes = len(unique_classes)
        self._binary_classifiers = []
        self._unique_classes = unique_classes

        for c in unique_classes:
            binary_y_train = (y_train == c).astype(int)
            clf = FisherBinaryClassifier()
            clf.fit(X_train, binary_y_train)
            self._binary_classifiers.append(clf)

        self._fitted = True

    def predict(self, X_test):
        """Predict labels for a collection of test examples.

        Args:
            X_test (list[list[float]]): Test examples to classify.
        Returns:
            list: Predicted labels for each test example.
        """
        if not self._fitted:
            raise ValueError("The model has not been fitted yet.")

        scores = []
        for clf in self._binary_classifiers:
            score = X_test @ clf.w - clf.threshold
            scores.append(score)
        predictions = self._unique_classes[np.argmax(scores, axis=0)]

        return predictions
