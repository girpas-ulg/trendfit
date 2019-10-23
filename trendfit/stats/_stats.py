import numpy as np

from ..base import BaseEstimator
from ..bootstrap import block_ar_wild
from ..models import LinearTrendFourier, LinearBrokenTrendFourier


class _BrokenTrendTest(BaseEstimator):
    """Implement the test as an estimator so that using it with
    a bootstrap algorithm is easy.

    """
    def __init__(self, model_trend, model_broken):
        self.model_trend = model_trend
        self.model_broken = model_broken

        self._parameters = {
            'ssr_trend': None,
            'ssr_broken': None,
            's_statistic': None,
        }

    def _fit(self, t, y):
        self.model_trend.fit(t, y)
        self.model_broken.fit(t, y)

        ssr_trend = sum(self.model_trend.residuals**2)
        ssr_broken = sum(self.model_broken.residuals**2)

        self._parameters['ssr_trend'] = ssr_trend
        self._parameters['ssr_broken'] = ssr_broken
        self._parameters['s_statistic'] = ssr_trend - ssr_broken

    def _predict(self, t):
        return self.model_trend.predict(t)


def broken_trend_test(t, y, f_order=3, alpha=0.05,
                      kw_model=None, kw_bootstrap=None):
    """Test of simple linear trend model vs. broken trend model.

    This is a formal test to determine whether or not the data
    suggests a sudden change in trend.

    Parameters
    ----------
    t : 1-D array
        Time coordinate.
    y : 1-D array
        Time series data.
    f_order : int, optional
        Finite order of the truncated Fourier series (default=3).
    alpha : float, optional
        Significance level of the test (default: 0.05).
    kw_model : dict, optional
        Keyword arguments passed to
        :class:`trendfit.models.LinearBrokenTrendFourier`
    kw_bootstrap : dict, optional
        Keyword arguments passed to
        :func:`trendfit.bootstrap.block_ar_wild`

    Returns
    -------
    s_statistic : float
        The value of the test statistic (see below). For high values,
        there is evidence that the model with break fits the data better.
    p_value : float
        P-value computed from the bootstrap estimated distribution.
    crit_value : float
        Critical value computed from the bootstrap estimated distribution
        given ``alpha``, which determines the cutoff point of the test.

    See Also
    --------
    :class:`trendfit.models.LinearTrendFourier`
    :class:`trendfit.models.LinearBrokenTrendFourier`

    Notes
    -----
    This test has the following pair of hypotheses:

    .. math::

       H_0 : \delta = 0\\
       H_1 : \delta \neq 0

    where :math:`\delta` is the change in trend slope.

    The test statistic is computed by subtracting the sum of squares
    of residuals of the fitted broken trend model to the sum of squares
    of residuals of the simple trend model.

    The location of the trend discontinuity (:math:`T_1`) is assumed
    unknown here.

    The P-value and the critical value are both obtained using autoregressive
    wild bootstrap. See [1]_ for more details.

    References
    ----------
    .. [1] M. Friedrich, E. Beutner, H. Reuvers, S. Smeekes, J-P. Urbain,
    W. Bader, B. Franco, B. Lejeune, and E. Mahieu, 2019. "Nonparametric
    estimation and bootstrap inference on trends in atmospheric time series:
    an application to ethane" arXiv:1903.05403v1

    """
    if kw_model is None:
        kw_m_broken = {}
    else:
        # those parameters must not be used in this test
        kw_m_broken = {k: v
                       for k, v in kw_model
                       if k not in ['t_break', 'opt_bounds']}

    m_trend = LinearTrendFourier(f_order=f_order)
    m_broken = LinearBrokenTrendFourier(f_order=f_order, **kw_m_broken)

    model = _BrokenTrendTest(m_trend, m_broken)
    model.fit(t, y)

    boot_res = block_ar_wild(model, **kw_bootstrap)

    s_statistic = model.parameters['s_statistic']
    s_dist = boot_res.parameter_dists['s_statistic']
    greater_s = s_dist > s_statistic

    # TODO: check if empty greater_s may really happen?
    if any(greater_s):
        p_value = np.mean(s_dist[greater_s])
    else:
        p_value = s_statistic

    crit_value = np.quantile(s_dist, 1 - alpha)

    return s_statistic, p_value, crit_value
