import pandas as pd


class MultivariateModelInspector:
    """Explain fitted multivariate regression models."""

    def __init__(self, experiment, feature_names):
        self.experiment = experiment
        self.feature_names = list(feature_names)

    def get_polynomial_details(self, model_name):
        """Return coefficients and the definitions of scaled inputs."""

        model = self.experiment.models[model_name]

        scaler = model.named_steps["scaler"]
        polynomial = model.named_steps["polynomial"]
        regression = model.named_steps["regression"]

        scaled_names = [
            f"z_{feature}"
            for feature in self.feature_names
        ]

        terms = polynomial.get_feature_names_out(
            scaled_names
        )

        coefficient_table = pd.DataFrame({
            "Term": ["Intercept"] + terms.tolist(),
            "Coefficient": (
                [float(regression.intercept_)]
                + regression.coef_.tolist()
            )
        })

        scaling_table = pd.DataFrame({
            "Feature": self.feature_names,
            "Scaled variable": scaled_names,
            "Training mean": scaler.mean_,
            "Training scale": scaler.scale_
        })

        return coefficient_table, scaling_table

    def get_complexity(self):
        """Describe the complexity of all six fitted models."""

        rows = []

        for name, model in self.experiment.models.items():
            if name.startswith("Polynomial"):
                polynomial = model.named_steps["polynomial"]
                regression = model.named_steps["regression"]

                number_of_terms = polynomial.n_output_features_

                details = (
                    f"{number_of_terms} polynomial terms; "
                    f"{number_of_terms + 1} coefficient slots "
                    f"including intercept; "
                    f"centered design rank = {regression.rank_}"
                )

            elif name.startswith("Decision tree"):
                details = (
                    f"Actual depth = {model.get_depth()}; "
                    f"nodes = {model.tree_.node_count}; "
                    f"leaves = {model.get_n_leaves()}"
                )

            else:
                regression = model.named_steps["regression"]

                details = (
                    f"k = {regression.n_neighbors}; "
                    "stores training examples; "
                    "no fitted regression coefficients"
                )

            rows.append({
                "Model": name,
                "Complexity": details
            })

        return pd.DataFrame(rows)