import copy
from collections import defaultdict

import numpy as np

from ..base import BaseEstimator
from ..options import OPTIONS


class BootstrapEstimator(BaseEstimator):
    """Base class for bootstrap algorithms."""

    def __init__(self, model, n_samples=1000, random_state=None,
                 save_models=False):
        """

        Parameters
        ----------
        model : object
            Any estimator, i.e., any object having the estimator interface
            (i.e., ``.fit`` and ``.predict`` methods, ``.parameters``
            and ``.residuals`` properties).
        n_samples: int, optional
            Number of bootstrap samples generated (default: 1000).
        random_state: int or object, optional
            Random seed or an instance of :class:`numpy.random.RandomState`
            used to generate the bootstrap samples, for reproducible
            experiments. If None (default), a new random state is defined.
            Note that this is ignored when running a bootstrap algorithm in
            parallel using dask.
        save_models: bool, optional
            If True, save all estimator instances created during the
            bootstrap run (default: False). This is useful, e.g., for
            access to the bootstrap sample after it has been run. This
            may consume a lot of memory!

        """
        self.model = model
        self.n_samples = n_samples
        self.save_models = save_models

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=random_state)

        super().__init__()

        self._parameter_dists = defaultdict(list)
        self._models = []

    @property
    def parameters(self):
        """Returns the (fitted) parameters of the estimator."""
        return self.model.parameters

    @property
    def parameter_dists(self):
        """Returns the bootstrap sampled distributions of
        the parameters of the estimator.

        """
        return self._parameter_dists

    @property
    def residuals(self):
        """Returns the estimator residuals."""
        return self.model.residuals

    @property
    def models(self):
        """Returns all estimator instances generated during
        bootstrap.

        Returns an empty list if ``save_models`` was set to False.
        """
        return self._models

    def generate_bootstrap_sample(self, random_state=None):
        if not self.model._fitted:
            raise ValueError("Model not fitted.")

        if random_state is None:
            random_state = self.random_state

        return self._generate_bootstrap_sample(random_state)

    def _generate_bootstrap_sample(self):
        raise NotImplementedError()

    def _fit(self, t, y):
        self.model.fit(t, y)

        def fit_sample(random_state=None):
            mb = copy.deepcopy(self.model)
            yb = self.generate_bootstrap_sample(random_state)

            mb.fit(t, yb)

            pb = mb.parameters.copy()

            if self.save_models:
                return mb, pb
            else:
                return None, pb

        if OPTIONS['use_dask']:
            import dask
            dlyd = [dask.delayed(fit_sample)(np.random.RandomState())
                    for _ in range(self.n_samples)]
            res = dask.compute(*dlyd)
        else:
            res = [fit_sample() for _ in range(self.n_samples)]

        for mb, pb in res:
            if self.save_models:
                self._models.append(mb)

            for k, v in pb.items():
                self._parameter_dists[k].append(v)

    def _predict(self, t):
        return self.model.predict(t)

    def get_ci_bounds(self, confidence_level=0.95):
        """Get the confidence intervals from bootstrap results.

        Parameters
        ----------
        confidence_level : float, optional
            Level of confidence (default: 0.95).

        Returns
        -------
        ci_bounds : dict of tuples
            The (lower, upper) bounds of the confidence interval
            for each parameter of the estimator.

        """
        if not self._fitted:
            raise ValueError("run `.fit()` first")

        ci_bounds = {}
        alpha = 1 - confidence_level

        for k, v in self.parameter_dists.items():
            lower = np.quantile(v, alpha / 2, axis=0)
            upper = np.quantile(v, 1 - alpha / 2, axis=0)

            ci_bounds[k] = (lower, upper)

        return ci_bounds


class ResidualResampling(BootstrapEstimator):
    """Residual Resampling Bootstrap.

    Generate bootstrap samples by (1) randomly resampling the
    residuals of the fitted model and (2) adding it to the predicted
    values.

    """
    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model : object
            Any estimator, i.e., any object having the estimator interface
            (i.e., ``.fit`` and ``.predict`` methods, ``.parameters``
            and ``.residuals`` properties).

        Other Parameters
        ----------------
        n_samples: int, optional
            Number of bootstrap samples generated (default: 1000).
        random_state: int or object, optional
            Random seed or an instance of :class:`numpy.random.RandomState`
            used to generate the bootstrap samples, for reproducible
            experiments. If None (default), a new random state is defined.
            Note that this is ignored when running a bootstrap algorithm in
            parallel using dask.
        save_models: bool, optional
            If True, save all estimator instances created during the
            bootstrap run (default: False). This is useful, e.g., for
            access to the bootstrap sample after it has been run. This
            may consume a lot of memory!

        """
        super().__init__(model, **kwargs)

    def _generate_bootstrap_sample(self, random_state):
        errors = self.model.residuals.copy()
        random_state.shuffle(errors)

        return self.model._y_predict + errors


def _cholesky_decomposition(t, gamma):
    mat = np.triu(gamma**(t[None, :] - t[:, None]))
    np.fill_diagonal(mat, 0.5)

    return np.linalg.cholesky(mat + mat.transpose())


class BlockARWild(BootstrapEstimator):
    """Block Autoregressive Wild Bootstrap.

    Generate bootstrap samples with autocorrelated errors using the
    method described in [1]_.

    This method may be used with non-evenly spaced samples.

    Notes
    -----

    Unlike the Autoregressive Wild Bootstrap method described in [1]_,
    the residuals are here split into contiguous blocks (equally
    sized, except maybe for the last block), and autocorrelated errors
    are generated independently for each of these blocks.

    One limitation of this approach is that the auto-correlation is
    "reset" each time when jumping from one block to another. However,
    in some cases this might be an acceptable approximation while
    offering great optimization in both speed-up and memory usage.
    Splitting the time-series in only a few number of blocks may
    result in 10x speed-up.

    To use the "full" Autoregressive Wild Bootstrap method presented
    in [1]_, just set a block size equal or larger than the actual
    size of the time series.

    The value of the autoregressive coefficient should tend to 1 as
    the size of the time series increases. In the absence of any given
    value, it will be set according to [2]_.

    References
    ----------
    .. [1] M. Friedrich, E. Beutner, H. Reuvers, S. Smeekes, J.-P. Urbain,
    W. Bader, B. Franco, B. Lejeune, and E. Mahieu, 2019. "Nonparametric
    estimation and bootstrap inference on trends in atmospheric time series:
    an application to ethane". arXiv:1903.05403v1

    ..[2] M. Friedrich, S. Smeekes, and J.-P. Urbain,
    2019. "Autoregressive wild bootstrap inference for nonparametric
    trends". Journal of Econometrics - Annals issue on econometric
    models of climate change, forthcoming

    """
    def __init__(self, model, ar_coef=None, block_size=500, **kwargs):
        """

        Parameters
        ----------
        model : object
            Any estimator, i.e., any object having the estimator interface
            (i.e., ``.fit`` and ``.predict`` methods, ``.parameters``
            and ``.residuals`` properties).
        ar_coef : float, optional
            autoregressive coefficient. If None (default), it will be set
            according to the size of the time-series.
        block_size : int, optional
            Size (number of samples) of the blocks (default: 500).

        Other Parameters
        ----------------
        n_samples: int, optional
            Number of bootstrap samples generated (default: 1000).
        random_state: int or object, optional
            Random seed or an instance of :class:`numpy.random.RandomState`
            used to generate the bootstrap samples, for reproducible
            experiments. If None (default), a new random state is defined.
            Note that this is ignored when running a bootstrap algorithm in
            parallel using dask.
        save_models: bool, optional
            If True, save all estimator instances created during the
            bootstrap run (default: False). This is useful, e.g., for
            access to the bootstrap sample after it has been run. This
            may consume a lot of memory!

        self.block_size = block_size
        self.ar_coef = ar_coef

        """
        self.ar_coef = ar_coef
        self.block_size = block_size

        super().__init__(model, **kwargs)

    def _generate_bootstrap_err(self, t, residuals, random_state):

        if self.ar_coef is None:
            # TODO: clarify this
            # this is not consistent with what is described in the paper
            # (in the paper, gamma, theta and l -> evenly spaced time-series)
            #l = 1.75 * t.size**(1/3)
            #gamma = 0.1**(1 / l)
            th = 0.01**(1 / (1.75 * t.size**(1/3)))
            l = 1 / 365.25
            gamma = th**(1. / l)
        else:
            gamma = self.ar_coef

        iid = random_state.normal(loc=0., scale=1., size=t.size)

        n_blocks = max(t.size // self.block_size, 1)
        t_blocks = np.array_split(t, n_blocks)
        residuals_blocks = np.array_split(residuals, n_blocks)
        iid_blocks = np.array_split(iid, n_blocks)

        def _gen_errors_block(tb, rb, iidb):
            L = _cholesky_decomposition(tb, gamma)
            return (L @ iidb).ravel() * rb

        return np.concatenate([
            _gen_errors_block(tb, rb, iidb)
            for tb, rb, iidb in zip(t_blocks, residuals_blocks, iid_blocks)
        ])

    def _generate_bootstrap_sample(self, random_state):
        errors = self._generate_bootstrap_err(self.model._t,
                                              self.model.residuals,
                                              random_state)

        return self.model._y_predict + errors
