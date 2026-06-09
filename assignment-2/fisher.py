import numpy as np


class FisherBinaryClassifier:
    """A simple Fisher's linear discriminant binary classifier.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X_train, y_train):
        """Fit the Fisher's linear discriminant model to the training data.

        Args:
            X_train (list[list[float]]): Training feature vectors.
            y_train (list): Training labels corresponding to each row in X_train.
        """
        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]

        # First, compute the means for each class
        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        # Next, compute the between class scatter matrices for each class
        d0 = (X0 - mu0).reshape(X0.shape[0], -1)
        S0 = d0.T @ d0
        d1 = (X1 - mu1).reshape(X1.shape[0], -1)
        S1 = d1.T @ d1

        # Calculate the total between-class scatter matrix
        SB = S0 + S1

        # Calculate w using the formula w = SB^-1 * (mu1 - mu0)
        self.w = np.linalg.pinv(SB) @ (mu1 - mu0)

        # Calculate the threshold as the midpoint between the projections of the class means onto w
        m0 = self.w @ mu0
        m1 = self.w @ mu1

        self.threshold = (m0 + m1) / 2
        self._fitted = True

    def predict(self, X_test):
        """Predict labels for a collection of test examples.
        Projects the test examples onto the weight vector w and compares
        to the threshold to determine class labels.

        Args:
            X_test (list[list[float]]): Test examples to classify.
        Returns:
            list: Predicted labels for each test example.
        """
        if not self._fitted:
            raise ValueError("The model has not been fitted yet.")

        # Project the test examples onto the weight vector w and compare to the threshold
        scores = X_test @ self.w - self.threshold

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
