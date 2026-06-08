class KNN:
    """A simple k-nearest neighbors classifier.

    Attributes:
        k (int): Number of nearest neighbors to consider during prediction.
        X_train (list[list[float]]): Training feature vectors.
        y_train (list): Training labels corresponding to each row in X_train.
    """

    def __init__(self, k):
        """Create a new KNN classifier.

        Args:
            k (int): Number of neighbors to use for majority voting.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """Store training data for later predictions.
        It is mainly used to avoid having to pass the training data
        to the predict method every time we want to make a prediction.

        Args:
            X_train (list[list[float]]): Training examples.
            y_train (list): Class labels for the training examples.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predict labels for a collection of test examples.

        Args:
            X_test (list[list[float]]): Test examples to classify.

        Returns:
            list: Predicted labels for each test example.
        """
        predicted_labels = [self._predict(x) for x in X_test]
        return predicted_labels

    def _predict(self, x):
        """Predict the label for a single test example.

        Args:
            x (list[float]): A single example to classify.

        Returns:
            The predicted label chosen by majority vote among the k nearest neighbors.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_nearest_neighbors = self._get_distances_with_idx_sorted(distances)
        k_nearest_labels = self._get_nearest_labels(k_nearest_neighbors)
        return self._get_most_voted_label(k_nearest_labels)

    def _get_distances_with_idx_sorted(self, distances):
        """Sort training examples by distance and return the closest k indices.

        Args:
            distances (list[float]): Distance values from a test example to each training example.

        Returns:
            list[tuple[int, float]]: Tuples of (index, distance) for the k closest neighbors.
        """
        distances_with_indices = [(index, distance) for index, distance in enumerate(distances)]
        sorted_distances = sorted(distances_with_indices, key=lambda x: x[1])
        return sorted_distances[:self.k]
    
    def _get_nearest_labels(self, distances_with_idx):
        """Lookup labels for the nearest neighbor indices.

        Args:
            distances_with_idx (list[tuple[int, float]]): Sorted index-distance pairs.

        Returns:
            list: Labels of the k nearest training examples.
        """
        k_nearest_labels = [self.y_train[index] for index, _ in distances_with_idx]
        return k_nearest_labels
    
    def _get_most_voted_label(self, labels):
        """Return the label with the highest vote count.
        This means the label that appears most frequently among the k nearest neighbors.
        

        Args:
            labels (list): Labels from the k nearest neighbors.

        Returns:
            The most common label among the given labels.
        """
        votes = {label: labels.count(label) for label in set(labels)}
        return max(votes, key=votes.get)

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two vectors.

        Args:
            x1 (list[float]): First vector.
            x2 (list[float]): Second vector.

        Returns:
            float: Euclidean distance between x1 and x2.
        """
        return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5
