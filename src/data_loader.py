from sklearn.datasets import load_diabetes


class DiabetesDataLoader:
    """Load and inspect the Scikit-Learn diabetes dataset."""

    def __init__(self, scaled=False):
        self.scaled = scaled

        self.dataset = None
        self.features = None
        self.target = None
        self.data = None

    def load_data(self):
        """Load the dataset and return a combined DataFrame."""

        self.dataset = load_diabetes(
            as_frame=True,
            scaled=self.scaled
        )

        self.features = self.dataset.data.copy()
        self.target = self.dataset.target.copy()
        self.target.name = "progression"

        self.data = self.features.copy()
        self.data["progression"] = self.target

        return self.data.copy()

    def _check_loaded(self):
        """Check that the dataset has been loaded."""

        if self.dataset is None:
            raise ValueError(
                "Load the dataset first by calling load_data()."
            )

    def get_description(self):
        """Return the official dataset description."""

        self._check_loaded()

        return self.dataset.DESCR

    def get_features_and_target(self):
        """Return separate copies of the input features and target."""

        self._check_loaded()

        return self.features.copy(), self.target.copy()

    def get_summary(self):
        """Return basic information about the loaded dataset."""

        self._check_loaded()

        return {
            "number_of_patients": len(self.data),
            "number_of_features": self.features.shape[1],
            "combined_shape": self.data.shape,
            "feature_names": self.features.columns.tolist(),
            "target_name": self.target.name,
            "target_minimum": self.target.min(),
            "target_maximum": self.target.max(),
            "scaled_features": self.scaled
        }