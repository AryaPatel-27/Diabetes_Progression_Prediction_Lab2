import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.model_evaluator import RegressionEvaluator


class MultivariateModelComparison:
    """Train and compare regression models using multiple features."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = None
        self.evaluator = RegressionEvaluator()

    @staticmethod
    def _polynomial_pipeline(degree):
        """Create a scaled polynomial regression pipeline."""

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
    def _knn_pipeline(neighbors):
        """Create a scaled kNN regression pipeline."""

        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "regression",
                KNeighborsRegressor(
                    n_neighbors=neighbors
                )
            )
        ])

    def _create_models(self):
        """Create the six model configurations."""

        return {
            "Polynomial degree 2": self._polynomial_pipeline(2),
            "Polynomial degree 3": self._polynomial_pipeline(3),

            "Decision tree depth 3": DecisionTreeRegressor(
                max_depth=3,
                random_state=self.random_state
            ),

            "Decision tree depth 5": DecisionTreeRegressor(
                max_depth=5,
                random_state=self.random_state
            ),

            "kNN k=5": self._knn_pipeline(5),
            "kNN k=15": self._knn_pipeline(15)
        }

    def train_and_compare(self, X_train, y_train, X_val, y_val):
        """Train each model and compare training/validation metrics."""

        self.models = self._create_models()
        rows = []

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)

            train_metrics = self.evaluator.evaluate(
                y_train,
                train_predictions,
                model_name=name
            ).iloc[0]

            val_metrics = self.evaluator.evaluate(
                y_val,
                val_predictions,
                model_name=name
            ).iloc[0]

            rows.append({
                "Model": name,
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        self.results = pd.DataFrame(rows)

        return self.results.copy()

    def predict(self, model_name, X):
        """Predict using one of the trained models."""

        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' has not been trained."
            )

        return self.models[model_name].predict(X)