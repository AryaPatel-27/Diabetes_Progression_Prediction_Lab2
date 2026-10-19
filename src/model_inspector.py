import pandas as pd


class PolynomialModelInspector:
    """Explain the fitted polynomial models."""

    def __init__(self, experiment):
        self.experiment = experiment

    def get_equation(self, degree):
        """Return the equation using standardized BMI."""

        model = self.experiment.models[degree]

        if degree == 0:
            constant = float(model.constant_.ravel()[0])

            return f"Predicted progression = {constant:.2f}"

        scaler = model.named_steps["scaler"]
        regression = model.named_steps["regression"]

        equation = (
            f"Predicted progression = {regression.intercept_:.2f}"
        )

        for power, coefficient in enumerate(
            regression.coef_,
            start=1
        ):
            sign = "+" if coefficient >= 0 else "-"
            term = "z" if power == 1 else f"z^{power}"

            equation += (
                f" {sign} {abs(coefficient):.2f}*{term}"
            )

        definition = (
            f"z = (BMI - {scaler.mean_[0]:.6f})"
            f" / {scaler.scale_[0]:.6f}"
        )

        return equation + "\n\n" + definition

    def get_parameter_counts(self):
        """Count regression coefficients and the intercept."""

        rows = []

        for degree, model in sorted(
            self.experiment.models.items()
        ):
            if degree == 0:
                terms = ["constant"]
                parameter_count = 1

            else:
                polynomial = model.named_steps["polynomial"]

                terms = polynomial.get_feature_names_out(
                    ["z"]
                ).tolist()

                parameter_count = len(terms) + 1

            rows.append({
                "Degree": degree,
                "Terms": ", ".join(terms),
                "Fitted regression parameters": parameter_count
            })

        return pd.DataFrame(rows)