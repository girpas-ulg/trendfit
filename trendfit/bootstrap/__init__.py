"""
The :mod:`trendfit.bootstrap` module gathers bootstrap algorithms
for estimating the uncertainty on time-series fitted model
parameters.

"""

from ._bootstrap import block_ar_wild, residual_resampling
