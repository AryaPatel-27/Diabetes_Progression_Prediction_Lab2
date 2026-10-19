import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticModelComparison:
    """Compare two classifiers for median-based progression categories."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.target_threshold = None
        self.models = {}
        self.results = None

    def create_labels(self, y):
        """Convert numerical progression scores into binary categories."""

        if self.target_threshold is None:
            raise ValueError("Train the models before creating labels.")

        return (y >= self.target_threshold).astype(int)

    @staticmethod
    def _calculate_metrics(actual, predicted, probabilities):
        """Calculate classification metrics."""

        auc = float("nan")

        if actual.nunique() == 2:
            auc = roc_auc_score(actual, probabilities)

        return {
            "Accuracy": accuracy_score(actual, predicted),
            "Precision": precision_score(
                actual,
                predicted,
                zero_division=0
            ),
            "Recall": recall_score(
                actual,
                predicted,
                zero_division=0
            ),
            "F1": f1_score(
                actual,
                predicted,
                zero_division=0
            ),
            "ROC AUC": auc
        }

    def predict(self, model_name, X):
        """Return predicted categories and class-1 probabilities."""

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been trained.")

        probabilities = self.models[
            model_name
        ].predict_proba(X)[:, 1]

        predicted_labels = (probabilities >= 0.5).astype(int)

        return predicted_labels, probabilities

    def train_and_compare(self, X_train, y_train, X_val, y_val):
        """Train two models and compare training/validation results."""

        # Learn the category boundary using training outcomes only
        self.target_threshold = float(y_train.median())

        train_labels = self.create_labels(y_train)
        val_labels = self.create_labels(y_val)

        self.models = {}
        rows = []

        for C in [0.1, 1.0]:
            model_name = f"Logistic Regression C={C}"

            model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        random_state=self.random_state
                    )
                )
            ])

            model.fit(X_train, train_labels)
            self.models[model_name] = model

            row = {
                "Model": model_name,
                "C": C
            }

            for split_name, X, actual_labels in [
                ("Train", X_train, train_labels),
                ("Validation", X_val, val_labels)
            ]:
                predicted, probabilities = self.predict(
                    model_name,
                    X
                )

                metrics = self._calculate_metrics(
                    actual_labels,
                    predicted,
                    probabilities
                )

                for metric_name, value in metrics.items():
                    row[f"{split_name} {metric_name}"] = value

            rows.append(row)

        self.results = pd.DataFrame(rows)

        return self.results.copy()

    def evaluate_test(self, model_name, X_test, y_test):
        """Evaluate a validation-selected model on test data."""

        actual_labels = self.create_labels(y_test)

        predicted, probabilities = self.predict(
            model_name,
            X_test
        )

        metrics = self._calculate_metrics(
            actual_labels,
            predicted,
            probabilities
        )

        results = pd.DataFrame([{
            "Model": model_name,
            **metrics
        }])

        matrix = pd.DataFrame(
            confusion_matrix(
                actual_labels,
                predicted,
                labels=[0, 1]
            ),
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        return results, matrix