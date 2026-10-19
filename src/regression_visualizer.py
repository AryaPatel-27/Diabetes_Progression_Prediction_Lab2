import matplotlib.pyplot as plt
import numpy as np


class RegressionComparisonVisualizer:
    """Compare regression predictions across dataset splits."""

    def plot_predictions(self, experiment, model_names, splits):
        """Show actual-versus-predicted plots for each model and split."""

        predictions = {}
        all_values = []

        for model_name in model_names:
            for split_name, (X, y) in splits.items():
                predicted = experiment.predict(
                    model_name=model_name,
                    X=X
                )

                predictions[(model_name, split_name)] = predicted

                all_values.extend([
                    np.asarray(y),
                    np.asarray(predicted)
                ])

        combined_values = np.concatenate(all_values)

        lower = combined_values.min()
        upper = combined_values.max()
        padding = (upper - lower) * 0.05

        limits = [
            lower - padding,
            upper + padding
        ]

        fig, axes = plt.subplots(
            len(model_names),
            len(splits),
            figsize=(15, 4 * len(model_names)),
            sharex=True,
            sharey=True,
            squeeze=False
        )

        for row, model_name in enumerate(model_names):
            for column, (split_name, (_, y)) in enumerate(
                splits.items()
            ):
                ax = axes[row, column]

                predicted = predictions[
                    (model_name, split_name)
                ]

                ax.scatter(
                    y,
                    predicted,
                    alpha=0.6
                )

                ax.plot(
                    limits,
                    limits,
                    color="black",
                    linestyle="--",
                    label="Perfect prediction"
                )

                ax.set_xlim(limits)
                ax.set_ylim(limits)

                ax.set_title(f"{model_name}\n{split_name}")
                ax.set_xlabel("Actual progression")
                ax.set_ylabel("Predicted progression")
                ax.grid(alpha=0.2)

        fig.suptitle(
            "Actual versus Predicted Progression",
            fontsize=16
        )

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()