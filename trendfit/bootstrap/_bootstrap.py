import copy
from collections import defaultdict

import numpy as np

from ..base import BaseEstimator


class BootstrapEstimator(BaseEstimator):

    def __init__(self, model, n_samples=1000, save_models=False):

        self.model = model
        self.n_samples = n_samples
        self.save_models = save_models

        super().__init__()

        self._parameter_dists = defaultdict(list)
        self._models = []

    @property
    def parameters(self):
        return self.model.parameters

    @property
    def parameter_dists(self):
        return self._parameter_dists

    @property
    def residuals(self):
        return self.model.residuals

    @property
    def models(self):
        return self._models

    def generate_bootstrap_sample(self):
        if not self.model._fitted:
            raise ValueError("Model not fitted.")

        return self._generate_bootstrap_sample()

    def _generate_bootstrap_sample(self):
        raise NotImplementedError()

    def _fit(self, t, y):
        self.model.fit(t, y)

        def fit_sample():
            mb = copy.deepcopy(self.model)
            yb = self.generate_bootstrap_sample()

            mb.fit(t, yb)

            return mb

        for b in range(self.n_samples):
            mb = fit_sample()

            for k, v in mb.parameters.items():
                self._parameter_dists[k].append(v)

            if self.save_models:
                self._models.append(mb)

    def _predict(self, t):
        return self.model.predict(t)

    def get_ci_bounds(self, confidence_level=0.95):
        if not self._fitted:
            raise ValueError("run `.fit()` first")

        ci_bounds = {}

        for k, v in self.parameter_dists.items():
            lower = np.quantile(v, 1 - confidence_level, axis=0)
            upper = np.quantile(v, confidence_level, axis=0)

            ci_bounds[k] = [lower, upper]

        return ci_bounds


def _cholesky_decomposition(t, gamma):
    mat = np.triu(gamma**(t[None, :] - t[:, None]))
    np.fill_diagonal(mat, 0.5)

    return np.linalg.cholesky(mat + mat.transpose())


def _gen_ar_wild_bootstrap_errors(t, residuals, gamma):
    L = _cholesky_decomposition(t, gamma)
    iid = np.random.normal(loc=0., scale=1., size=t.size)[:, None]

    return (L @ iid).ravel() * residuals


class BlockARWild(BootstrapEstimator):
    """Block Autoregressive Wild Bootstrap.

    Generate bootstrap samples with autocorrelated errors using the
    method described in [add paper], which may be used with non-evenly
    spaced samples.

    Unlike the Autoregressive Wild Bootstrap method described in [add
    paper], this method is applied on successive subsets (blocks) of
    the time series.

    One limitation with this approach is that the autoregression is
    "reset" each time when jumping from one block to another. However,
    in some cases this might be an acceptable approximation while
    providing great optimization in both speed-up and memory usage
    (the overall matrix size rapidly decreases when increasing the
    number of blocks).

    To use the "classic" Autoregressive Wild Bootstrap method, just
    set a block size equal or larger than the size of the time series.

    """

    def __init__(self, model, n_samples=1000, block_size=500, ar_coef=None,
                 save_models=False):

        self.block_size = block_size
        self.ar_coef = ar_coef

        super().__init__(model, n_samples, save_models)

    def _generate_bootstrap_err(self, t, residuals):
        # autoregressive coefficient
        if self.ar_coef is None:
            th = 0.01**(1 / (1.75 * t.size**(1/3)))
            l = 1 / 365.25
            gamma = th**(1. / l)
        else:
            gamma = self.ar_coef

        n_blocks = max(t.size // self.block_size, 1)
        t_blocks = np.array_split(t, n_blocks)
        residuals_blocks = np.array_split(residuals, n_blocks)

        return np.concatenate([
            _gen_ar_wild_bootstrap_errors(tb, rb, gamma)
            for tb, rb in zip(t_blocks, residuals_blocks)
        ])

    def _generate_bootstrap_sample(self):
        errors = self._generate_bootstrap_err(self.model._t,
                                              self.model.residuals)

        return self.model._y_predict + errors
