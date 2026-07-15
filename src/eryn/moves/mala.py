# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cholesky
from eryn.moves.mh import MHMove

__all__ = ["MALAMove"]

class MALAMove(MHMove):
    def __init__(
        self,
        grad_all,
        epsilon_all=dict(),
        constant_metric=False,
        metric=None,
        vectorized=False,
        indices=dict(),
        **kwargs,
    ):
        self.epsilon = dict()
        self.grad_function = dict()
        self.metric = dict()
        self.L = dict()  # FIX: was missing
        self.constant_metric = constant_metric
        self.indices = dict()
        self.vectorized = vectorized  # FIX: was missing

        names = list(grad_all.keys())
        for name in names:
            self.indices[name] = indices.get(name, None)
            self.grad_function[name] = grad_all[name]
            self.epsilon[name] = epsilon_all.get(name, 1.0)
            if self.constant_metric:
                if metric is None or name not in metric:
                    raise ValueError("Specify the metric if using constant metric")
                self.metric[name] = metric[name]
                self.L[name] = cholesky((metric[name] + metric[name].T) / 2, lower=True)

        super().__init__(**kwargs)

    def _log_proposal_pdf(self, x, mu, epsilon, L):
        """Log pdf of MALA proposal: N(mu + gradU, epsilon^2 * M)"""
        diff = x - mu  # (n_active, ndim_subset)
        z = np.linalg.solve(L, diff.T)  # (ndim_subset, n_active)
        ndim = L.shape[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (np.sum(z**2, axis=0) / epsilon**2 + ndim * np.log(2 * np.pi)
                       + ndim * 2 * np.log(epsilon) + log_det)  # (n_active,)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        q = {}
        first_name = list(branches_coords.keys())[0]
        ntemps, nwalkers, _, _ = branches_coords[first_name].shape
        factors = np.zeros((ntemps, nwalkers), dtype=np.float64)

        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            eps = self.epsilon[name]

            if self.indices[name] is None:
                self.indices[name] = np.arange(ndim)
            idx = self.indices[name]

            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            inds_here = np.where(inds)

            q[name] = coords.copy()
            new_coords = coords.copy()

            # --- Gradients and metric at x ---
            if self.vectorized:
                gradients, fishers = self.grad_function[name](coords[inds_here])
            else:
                tmp = [self.grad_function[name](c) for c in coords[inds_here]]
                gradients = np.array([t[0] for t in tmp])
                fishers = np.array([t[1] for t in tmp])

            if self.constant_metric:
                M = self.metric[name]
                L = self.L[name]
            else:
                metrics = np.linalg.inv(fishers)
                # symmetrize
                metrics = (metrics + metrics.transpose(0, 2, 1)) / 2

            # --- Proposal x -> y ---
            coords_active = coords[inds_here][:, idx]  # (n_active, ndim_subset)

            if self.constant_metric:
                gradU = 0.5 * eps**2 * gradients[:, idx] @ M.T
                noise = eps * random.randn(*coords_active.shape) @ L.T
            else:
                gradU = 0.5 * eps**2 * np.einsum('bi,bij->bj', gradients[:, idx], metrics)
                # per-sample cholesky
                L_arr = []
                for m in metrics:
                    #print(m)
                    L_arr.append(cholesky((m + m.T) / 2, lower=True))
                L_arr = np.asarray(L_arr)
                #L_arr = np.array([cholesky((m + m.T) / 2, lower=True) for m in metrics])
                z = random.randn(*coords_active.shape)
                noise = eps * np.einsum('bij,bj->bi', L_arr, z)

            y_active = coords_active + gradU + noise# Instead of:

            # Do:
            active = new_coords[inds_here]
            active[:, idx] = y_active
            new_coords[inds_here] = active
            #print(gradU + noise)
            # --- Gradients and metric at y ---
            if self.vectorized:
                gradients_y, fishers_y = self.grad_function[name](new_coords[inds_here])
            else:
                tmp = [self.grad_function[name](c) for c in new_coords[inds_here]]
                gradients_y = np.array([t[0] for t in tmp])
                fishers_y = np.array([t[1] for t in tmp])

            if self.constant_metric:
                gradU_y = 0.5 * eps**2 * M @ gradients_y[:, idx]
                L_y = L
            else:
                metrics_y = np.linalg.inv(fishers_y)
                metrics_y = (metrics_y + metrics_y.transpose(0, 2, 1)) / 2
                gradU_y = 0.5 * eps**2 * np.einsum('bi,bij->bj', gradients_y[:, idx], metrics_y)
                L_y = np.array([cholesky((m + m.T) / 2, lower=True) for m in metrics_y])

            # --- Proposal log weights: log q(x|y) - log q(y|x) ---
            # mean of forward proposal q(y|x)
            mu_fwd = coords_active + gradU           # (n_active, ndim_subset)
            # mean of reverse proposal q(x|y)
            mu_rev = y_active + gradU_y              # (n_active, ndim_subset)

            if self.constant_metric:
                log_q_fwd = self._log_proposal_pdf(y_active, mu_fwd, eps, L)
                log_q_rev = self._log_proposal_pdf(coords_active, mu_rev, eps, L_y)
            else:
                log_q_fwd = np.array([
                    self._log_proposal_pdf(y_active[i:i+1], mu_fwd[i:i+1], eps, L_arr[i])
                    for i in range(len(y_active))
                ]).squeeze()
                log_q_rev = np.array([
                    self._log_proposal_pdf(coords_active[i:i+1], mu_rev[i:i+1], eps, L_y[i])
                    for i in range(len(y_active))
                ]).squeeze()

            # accumulate factor: log q(x|y) - log q(y|x)
            factors[inds_here[:2]] += log_q_rev - log_q_fwd

            q[name][inds_here] = new_coords[inds_here]

        return q, factors
