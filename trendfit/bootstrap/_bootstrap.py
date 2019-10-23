import copy
from collections import defaultdict

import numpy as np

from ..base import BaseEstimator
from ..options import OPTIONS


class BootstrapRunner:

    def __init__(self, model, n_samples=1000, random_state=None,
                 save_models=False):
        if not model._fitted:
            raise ValueError("Model is not fitted. Run `.fit()` first.")

        self.model = model
        self.n_samples = n_samples
        self.save_models = save_models

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=random_state)

        self.parameter_dists = defaultdict(list)
        self.models = []

    def generate_sample(self, random_state):
        raise NotImplementedError()

    def run(self):

        def fit_sample(random_state=None):
            mb = copy.deepcopy(self.model)
            yb = self.generate_sample(random_state)

            mb.fit(self.model._t, yb)

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
            res = [fit_sample(self.random_state)
                   for _ in range(self.n_samples)]

        for mb, pb in res:
            if self.save_models:
                self.models.append(mb)

            for k, v in pb.items():
                self.parameter_dists[k].append(v)


class BootstrapResults:
    """Store bootstrap results and compute statistics such as confidence
    intervals.

    """
    def __init__(self, runner):
        self._runner = runner

    @property
    def parameter_dists(self):
        """Returns the bootstrap sampled distributions of
        the parameters of the estimator.

        """
        return self._runner.parameter_dists

    @property
    def models(self):
        """Returns all estimator instances generated during
        bootstrap.

        Returns an empty list if ``save_models`` was set to False.
        """
        return self._runner.models

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
        ci_bounds = {}
        alpha = 1 - confidence_level

        for k, v in self.parameter_dists.items():
            lower = np.quantile(v, alpha / 2, axis=0)
            upper = np.quantile(v, 1 - alpha / 2, axis=0)

            ci_bounds[k] = (lower, upper)

        return ci_bounds


class ResidualResamplingRunner(BootstrapRunner):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def generate_sample(self, random_state):
        errors = self.model.residuals.copy()
        random_state.shuffle(errors)

        return self.model._y_predict + errors


def residual_resampling(model, n_samples=1000, **kwargs):
    """Residual Resampling Bootstrap.

    Generate bootstrap samples by (1) randomly resampling the
    residuals of the fitted model and (2) adding it to the predicted
    values.

    Parameters
    ----------
    model : object
        Any estimator, i.e., any object having the estimator interface
        (i.e., ``.fit`` and ``.predict`` methods, ``.parameters``
        and ``.residuals`` properties).
    n_samples: int, optional
        Number of bootstrap samples generated (default: 1000).

    Other Parameters
    ----------------
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
    runner = ResidualResamplingRunner(model, n_samples=n_samples, **kwargs)

    runner.run()

    return BootstrapResults(runner)


def _cholesky_decomposition(t, gamma):
    mat = np.triu(gamma**(t[None, :] - t[:, None]))
    np.fill_diagonal(mat, 0.5)

    return np.linalg.cholesky(mat + mat.transpose())


class BlockARWildRunner(BootstrapRunner):

    def __init__(self, model, ar_coef=None, block_size=500, use_cache=False,
                 **kwargs):
        self.ar_coef = ar_coef
        self.block_size = block_size
        self.use_cache = use_cache

        super().__init__(model, **kwargs)

    def _block_cholesky_decomp(self):
        t = self.model._t

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

        t_blocks = np.array_split(t, self._n_blocks)

        return [_cholesky_decomposition(tb, gamma) for tb in t_blocks]

    def _generate_sample_err(self, residuals, random_state):
        if self.use_cache:
            l_blocks = self._l_blocks
        else:
            l_blocks = self._block_cholesky_decomp()

        iid = random_state.normal(loc=0., scale=1.,
                                  size=self.model._t.size)

        r_blocks = np.array_split(residuals, self._n_blocks)
        iid_blocks = np.array_split(iid, self._n_blocks)

        return np.concatenate([
            (lb @ iidb).ravel() * rb
            for lb, rb, iidb in zip(l_blocks, r_blocks, iid_blocks)
        ])

    def generate_sample(self, random_state):
        errors = self._generate_sample_err(self.model.residuals,
                                           random_state)

        return self.model._y_predict + errors

    def run(self):
        self._n_blocks = max(self.model._t.size // self.block_size, 1)

        if self.use_cache:
            self._l_blocks = self._block_cholesky_decomp()

        super().run()


def block_ar_wild(model, ar_coef=None, block_size=500, n_samples=1000,
                  use_cache=False, **kwargs):
    """Block Autoregressive Wild Bootstrap.

    Generate bootstrap samples with autocorrelated errors using the
    method described in [1]_.

    This method may be used with non-evenly spaced samples.

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
    n_samples : int, optional
        Number of bootstrap samples generated (default: 1000)
    use_cache : bool, optional
        If True, eagerly compute Cholesky decomposition for generating
        autocorrelated samples and cache the results in memory.
        (default: False). Beware that if parallel computation is enabled
        with dask (depending on which scheduler is used) this might be highly
        memory expensive!!

    Other Parameters
    ----------------
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
    runner = BlockARWildRunner(model, ar_coef=ar_coef, block_size=block_size,
                               n_samples=n_samples, use_cache=use_cache,
                               **kwargs)

    runner.run()

    return BootstrapResults(runner)
