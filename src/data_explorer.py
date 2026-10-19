import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DiabetesDataExplorer:
    """Explore the diabetes dataset using tables and plots."""

    def __init__(self, data, target_column="progression"):
        self.data = data.copy()
        self.target_column = target_column

        if self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' was not found."
            )

        self.feature_columns = [
            column
            for column in self.data.columns
            if column != self.target_column
        ]

    def get_data_overview(self):
        """Return column types, missing counts, and unique values."""

        return pd.DataFrame({
            "data_type": self.data.dtypes.astype(str),
            "non_missing_values": self.data.notna().sum(),
            "missing_values": self.data.isna().sum(),
            "unique_values": self.data.nunique()
        })

    def get_statistics(self):
        """Return descriptive statistics for numeric columns."""

        return self.data.describe().T

    def count_duplicates(self):
        """Count fully duplicated rows, excluding the index."""

        return int(self.data.duplicated().sum())

    def plot_histograms(self):
        """Show the distribution of each column."""

        self.data.hist(
            bins=20,
            figsize=(15, 12),
            color="steelblue",
            edgecolor="white"
        )

        plt.suptitle(
            "Distributions of Patient Measurements and Progression",
            fontsize=16
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def plot_scatter_plots(self):
        """Plot each input feature against the target."""

        number_of_columns = 3

        number_of_rows = math.ceil(
            len(self.feature_columns) / number_of_columns
        )

        fig, axes = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(15, 4 * number_of_rows),
            squeeze=False
        )

        axes = axes.ravel()

        for index, feature in enumerate(self.feature_columns):
            sns.scatterplot(
                data=self.data,
                x=feature,
                y=self.target_column,
                ax=axes[index],
                alpha=0.6
            )

            axes[index].set_title(
                f"{feature} versus progression"
            )

            axes[index].set_ylabel("Progression score")

        for index in range(len(self.feature_columns), len(axes)):
            axes[index].set_visible(False)

        fig.suptitle(
            "Baseline Features versus Disease Progression",
            fontsize=16
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def get_correlation_matrix(self):
        """Return Pearson correlations between numeric columns."""

        return self.data.corr(numeric_only=True)

    def plot_correlation_matrix(self):
        """Display the correlation matrix as a heatmap."""

        correlations = self.get_correlation_matrix()

        plt.figure(figsize=(12, 9))

        sns.heatmap(
            correlations,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1
        )

        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def get_target_correlations(self):
        """Order features by absolute correlation with the target."""

        correlations = self.get_correlation_matrix()[
            self.target_column
        ].drop(self.target_column)

        ordered_features = (
            correlations.abs()
            .sort_values(ascending=False)
            .index
        )

        return correlations.loc[ordered_features].to_frame(
            name="correlation_with_progression"
        )