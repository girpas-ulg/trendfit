"""Base classes."""


class BaseEstimator:
    """Base class for all estimators."""

    def __init__(self):
        self._parameters = {}
        self._residuals = None
        self._fitted = False
        self._t = None
        self._y = None
        self._y_predict = None

    @property
    def parameters(self):
        """Returns the model (fitted) parameters."""
        return self._parameters

    def fit(self, t, y):
        """Fit the model to time-series data.

        Parameters
        ----------
        t : 1-D array
            Time coordinate.
        y : 1-D array
            Time series data.

        """
        res = self._fit(t, y)

        self._fitted = True
        self._t = t
        self._y = y
        self._y_predict = self.predict(t)
        self._residuals = y - self._y_predict

        return res

    def _fit(self, t, y):
        raise NotImplementedError()

    def predict(self, t):
        """Evaluate the model at given time values.

        Parameters
        ----------
        t : 1-D array
            Time coordinate.

        Returns
        -------
        y_predicted : 1-D array
            Predicted values.

        """
        if not self._fitted:
            raise ValueError("run `.fit()` first")

        return self._predict(t)

    def _predict(self, t):
        raise NotImplementedError()

    @property
    def residuals(self):
        """Returns the model residuals."""
        return self._residuals
