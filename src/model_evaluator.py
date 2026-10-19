import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)


class RegressionEvaluator:
    """Evaluate numerical predictions using regression metrics."""

    def evaluate(self, y_true, y_pred, model_name):
        """Return a one-row table of evaluation results."""

        return pd.DataFrame([{
            "Model": model_name,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": (
                mean_absolute_percentage_error(y_true, y_pred) * 100
            )
        }])