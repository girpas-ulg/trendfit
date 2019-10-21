"""Base classes."""


class BaseEstimator:

    def __init__(self):
        self._parameters = {}
        self._residuals = None
        self._fitted = False
        self._t = None
        self._y = None
        self._y_predict = None

    @property
    def parameters(self):
        return self._parameters

    def fit(self, t, y):
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
        if not self._fitted:
            raise ValueError("run `.fit()` first")

        return self._predict(t)

    def _predict(self, t):
        raise NotImplementedError()

    @property
    def residuals(self):
        return self._residuals
