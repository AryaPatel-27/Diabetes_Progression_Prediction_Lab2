import matplotlib.pyplot as plt
import pandas as pd


class RegressionErrorAnalyzer:
    """Inspect prediction errors for a regression model."""

    def __init__(self, X, y_true, y_pred):
        self.errors = X[["bmi"]].copy()

        self.errors["actual"] = y_true
        self.errors["predicted"] = y_pred

        self.errors["residual"] = (
            self.errors["actual"] - self.errors["predicted"]
        )

        self.errors["absolute_error"] = (
            self.errors["residual"].abs()
        )

    def get_largest_errors(self, number=10):
        """Return patients with the largest absolute errors."""

        return self.errors.sort_values(
            "absolute_error",
            ascending=False
        ).head(number)

    def get_group_summary(self, y_train):
        """Summarize errors using training-target quartiles."""

        lower = y_train.quantile(0.25)
        upper = y_train.quantile(0.75)

        grouped_data = self.errors.copy()

        grouped_data["progression_group"] = pd.cut(
            grouped_data["actual"],
            bins=[float("-inf"), lower, upper, float("inf")],
            labels=["Lower", "Middle", "Higher"]
        )

        return grouped_data.groupby(
            "progression_group",
            observed=True
        ).agg(
            patients=("actual", "size"),
            MAE=("absolute_error", "mean"),
            mean_residual=("residual", "mean")
        )

    def plot_residuals(self):
        """Plot residuals against predicted progression."""

        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.errors["predicted"],
            self.errors["residual"],
            alpha=0.7
        )

        plt.axhline(
            0,
            color="black",
            linestyle="--"
        )

        plt.xlabel("Predicted progression score")
        plt.ylabel("Residual: actual − predicted")
        plt.title("Test-Set Prediction Errors")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()