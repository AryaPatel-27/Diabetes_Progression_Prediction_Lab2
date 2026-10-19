import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class UnivariatePolynomialRegression:
    """Train and compare polynomial regression models using one feature."""

    def __init__(self, feature="bmi", max_degree=5):
        self.feature = feature
        self.max_degree = max_degree
        self.models = {}
        self.results = None

    def _create_model(self, degree):
        """Create a constant baseline or a polynomial pipeline."""

        if degree == 0:
            return DummyRegressor(strategy="mean")

        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "polynomial",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),
            ("regression", LinearRegression())
        ])

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Calculate the required regression metrics."""

        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": (
                mean_absolute_percentage_error(y_true, y_pred) * 100
            )
        }

    def train_and_compare(self, X_train, y_train, X_val, y_val):
        """Fit models on training data and evaluate both sets."""

        train_features = X_train[[self.feature]]
        val_features = X_val[[self.feature]]

        self.models = {}
        rows = []

        for degree in range(self.max_degree + 1):
            model = self._create_model(degree)

            model.fit(train_features, y_train)

            train_predictions = model.predict(train_features)
            val_predictions = model.predict(val_features)

            train_metrics = self._calculate_metrics(
                y_train,
                train_predictions
            )

            val_metrics = self._calculate_metrics(
                y_val,
                val_predictions
            )

            self.models[degree] = model

            rows.append({
                "Degree": degree,
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        self.results = pd.DataFrame(rows)

        return self.results.copy()

    def predict(self, degree, X):
        """Predict using a previously trained model."""

        if degree not in self.models:
            raise ValueError(
                f"Degree {degree} has not been trained yet."
            )

        return self.models[degree].predict(X[[self.feature]])