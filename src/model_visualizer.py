import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PolynomialModelVisualizer:
    """Plot a trained polynomial model against each dataset split."""

    def __init__(self, experiment, degree):
        self.experiment = experiment
        self.degree = degree
        self.feature = experiment.feature

    def plot_fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ):
        """Show the same trained curve in three comparison panels."""

        splits = [
            ("Training", X_train, y_train, "steelblue"),
            ("Validation", X_val, y_val, "darkorange"),
            ("Test", X_test, y_test, "seagreen")
        ]

        # Use one shared BMI range for all three panels
        all_bmi = pd.concat([
            X_train[self.feature],
            X_val[self.feature],
            X_test[self.feature]
        ])

        bmi_grid = pd.DataFrame({
            self.feature: np.linspace(
                all_bmi.min(),
                all_bmi.max(),
                400
            )
        })

        curve_predictions = self.experiment.predict(
            degree=self.degree,
            X=bmi_grid
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5),
            sharex=True,
            sharey=True
        )

        for ax, (name, X, y, color) in zip(axes, splits):
            ax.scatter(
                X[self.feature],
                y,
                color=color,
                alpha=0.6,
                label=f"{name} patients"
            )

            ax.plot(
                bmi_grid[self.feature],
                curve_predictions,
                color="black",
                linewidth=2,
                label=f"Degree-{self.degree} model"
            )

            ax.set_title(f"{name} Set")
            ax.set_xlabel("BMI")
            ax.grid(alpha=0.2)
            ax.legend()

        axes[0].set_ylabel("Progression score")

        fig.suptitle(
            "Selected Polynomial Model: BMI versus Progression"
        )

        fig.tight_layout()
        plt.show()