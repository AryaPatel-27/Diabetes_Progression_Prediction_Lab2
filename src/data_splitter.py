from sklearn.model_selection import train_test_split


class DiabetesDataSplitter:
    """Split data into training, validation, and test sets."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def split(self, X, y):
        """Return approximately 75% train, 10% validation, 15% test."""

        # Keep 75% for training and 25% for the remaining sets
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X,
            y,
            train_size=0.75,
            random_state=self.random_state
        )

        # Divide the remaining 25%:
        # 40% becomes validation (10% of the original data)
        # 60% becomes test (15% of the original data)
        X_val, X_test, y_val, y_test = train_test_split(
            X_remaining,
            y_remaining,
            test_size=0.60,
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test