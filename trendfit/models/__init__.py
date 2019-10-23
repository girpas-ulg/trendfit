"""
The :mod:`trendfit.models` module gathers models for fitting
time-series data.

"""

from ._models import (LinearNoTrendFourier,
                      LinearTrendFourier,
                      LinearBrokenTrendFourier)
