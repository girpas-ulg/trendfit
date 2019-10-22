import numpy as np
from scipy.optimize import dual_annealing

from ..base import BaseEstimator


class LinearTrendFourier(BaseEstimator):

    def __init__(self, ndegrees=3):
        self.ndegrees = ndegrees

        self._parameters = {
            'intercept': None,
            'trend': None,
            'fourier_terms': [],
        }

        super().__init__()

    def _fourier_terms(self, t, degree):
        return [np.cos(2 * degree * np.pi * t),
                np.sin(2 * degree * np.pi * t)]

    def _regressor_terms(self, t):
        # intercept, trend
        reg_terms = [np.ones(t.size), t]

        # fourier terms
        for degree in range(1, self.ndegrees + 1):
            reg_terms += self._fourier_terms(t, degree)

        reg_idx = {
            'intercept': 0,
            'trend': 1,
            'fourier_terms': slice(2, None)
        }

        return reg_idx, reg_terms

    def _solve_lstsq(self, t, y, reg_idx, reg_terms):
        mat = np.stack(reg_terms).transpose()

        p, ssr, _, _ = np.linalg.lstsq(mat, y, rcond=None)

        for k, idx in reg_idx.items():
            self._parameters[k] = p[idx]

        return ssr

    def _fit(self, t, y):
        reg_idx, reg_terms = self._regressor_terms(t)

        return self._solve_lstsq(t, y, reg_idx, reg_terms)[0]

    def _compute_y(self, t, reg_idx, reg_terms):
        p = np.empty((len(reg_terms)))

        for k, idx in reg_idx.items():
            p[idx] = self._parameters[k]

        mat = np.stack(reg_terms).transpose()

        return (mat @ p[:, None]).ravel()

    def _predict(self, t):
        reg_idx, reg_terms = self._regressor_terms(t)

        return self._compute_y(t, reg_idx, reg_terms)


class DualLinearTrendFourier(LinearTrendFourier):

    def __init__(self, ndegrees=3, t_break=None, t0=None):
        super().__init__(ndegrees)

        if t0 is not None:
            self.t0 = [t0]
        else:
            self.t0 = None

        self._fit_t_break = t_break is None
        self._parameters['t_break'] = t_break

    def _regressor_terms(self, t, t_break):
        reg_idx, reg_terms = super()._regressor_terms(t)

        reg_terms.append(np.where(t > t_break, t - t_break, 0.))
        reg_idx['trend_change'] = -1

        return reg_idx, reg_terms

    def _fit(self, t, y):

        def solve_for_location(t_break):
            # solve system with a-priori t_break value

            reg_idx, reg_terms = self._regressor_terms(t, t_break)

            ssr = self._solve_lstsq(t, y, reg_idx, reg_terms)

            # system solving issues with t_break near bounds
            if not len(ssr):
                return np.inf
            else:
                return ssr[0]

        if self._fit_t_break:
            res = dual_annealing(solve_for_location,
                                 [(t[1], t[-1])],
                                 x0=self.t0,
                                 maxiter=500)

            self._parameters['t_break'] = res.x[0]

        # rerun lstsq to properly set other parameter values
        reg_idx, reg_terms = self._regressor_terms(
            t, self._parameters['t_break']
        )
        res_lstsq = self._solve_lstsq(t, y, reg_idx, reg_terms)

        if self._fit_t_break:
            return res
        else:
            return res_lstsq

    def _predict(self, t):
        reg_idx, reg_terms = self._regressor_terms(
            t, self._parameters['t_break']
        )

        return self._compute_y(t, reg_idx, reg_terms)
