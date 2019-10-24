"""
The :mod:`trendfit.models` module gathers models for fitting
time-series data.

"""

from ._models import (KernelTrend,
                      LinearNoTrendFourier,
                      LinearTrendFourier,
                      LinearBrokenTrendFourier)
