import numpy as np

from ..base import BaseEstimator
from ..bootstrap import BlockARWild
from ..models import LinearTrendFourier, LinearBrokenTrendFourier


class _BrokenTrendTest(BaseEstimator):
    """Implement the test as an estimator in order to easily wrap
    it in bootstrap classes.

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

    if kw_model is None:
        kw_m_broken = {}
    else:
        # those parameters must not be used in this test
        kw_m_broken = {k: v
                       for k, v in kw_model
                       if k not in ['t_break', 'opt_bounds']}

    m_trend = LinearTrendFourier(f_order=f_order)
    m_broken = LinearBrokenTrendFourier(f_order=f_order, **kw_m_broken)

    model = BlockARWild(_BrokenTrendTest(m_trend, m_broken),
                        **kw_bootstrap)

    model.fit(t, y)

    s_statistic = model.parameters['s_statistic']

    s_dist = model.parameter_dists['s_statistic']
    greater_s = s_dist > s_statistic

    if any(greater_s):
        p_value = np.mean(s_dist[greater_s])
    else:
        p_value = s_statistic

    crit_value = np.quantile(s_dist, 1 - alpha)

    return s_statistic, p_value, crit_value
