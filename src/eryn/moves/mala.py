# -*- coding: utf-8 -*-
import numpy as np

from copy import deepcopy
from eryn.state import State

from eryn.moves.move import Move
from scipy.linalg import cholesky

__all__ = ["MALAMove"]

class MALAMove(Move):
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
        self.alpha = np.array([])

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

    def setup(self, branches_coords):
        """Any setup for the proposal.

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """


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

            if self.constant_metric:

                if self.vectorized:
                    gradients = self.grad_function[name](coords[inds_here])
                else:
                    gradients = np.asarray([self.grad_function[name](c) for c in coords[inds_here]])
                M = self.metric[name]
                L = self.L[name]
            else:
                if self.vectorized:
                    gradients, fishers = self.grad_function[name](coords[inds_here])
                else:
                    tmp = [self.grad_function[name](c) for c in coords[inds_here]]
                    gradients = np.array([t[0] for t in tmp])
                    fishers = np.array([t[1] for t in tmp])
                metrics = np.linalg.inv(fishers)
                # symmetrize
                metrics = (metrics + metrics.transpose(0, 2, 1)) / 2

            # --- Proposal x -> y ---
            coords_active = coords[inds_here][:, idx]  # (n_active, ndim_subset)

            if self.constant_metric:

                gradU = 0.5 * eps**2 *  gradients @ M.T
                noise = eps * random.randn(*coords_active.shape) @ L.T

            else:
                gradU = 0.5 * eps**2 * np.einsum('bi,bij->bj', gradients, metrics)
                # per-sample cholesky
                L_arr = [cholesky((m + m.T) / 2, lower=True) for m in metrics]
                L_arr = np.asarray(L_arr)
                #L_arr = np.array([cholesky((m + m.T) / 2, lower=True) for m in metrics])
                z = random.randn(*coords_active.shape)
                noise = eps * np.einsum('bij,bj->bi', L_arr, z)

            y_active = coords_active + gradU + noise

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

                if self.vectorized:
                    gradients_y= self.grad_function[name](new_coords[inds_here])
                else:
                    gradients_y = np.array([self.grad_function[name](c) for c in new_coords[inds_here]])
                gradU_y = 0.5 * eps**2 * gradients_y@  M.T
                L_y = L
            else:
                if self.vectorized:
                    gradients_y, fishers_y = self.grad_function[name](new_coords[inds_here])
                else:
                    tmp = [self.grad_function[name](c) for c in new_coords[inds_here]]
                    gradients_y = np.array([t[0] for t in tmp])
                    fishers_y = np.array([t[1] for t in tmp])


                metrics_y = np.linalg.inv(fishers_y)
                metrics_y = (metrics_y + metrics_y.transpose(0, 2, 1)) / 2
                gradU_y = 0.5 * eps**2 * np.einsum('bi,bij->bj', gradients_y, metrics_y)
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

    
    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """

        self.setup(state.branches_coords)

        # get all branch names for gibbs setup
        all_branch_names = list(state.branches.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # in case there are no leaves yet
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # iterate through gibbs setup
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # setup supplemental information
            if not np.all(
                np.asarray(list(state.branches_supplemental.values())) == None
            ):
                new_branch_supps = deepcopy(state.branches_supplemental)
            else:
                new_branch_supps = None

            if state.supplemental is not None:
                new_supps = deepcopy(state.supplemental)
            else:
                new_supps = None

            # setup information according to gibbs info
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            # if no walkers are actually being proposed
            if not at_least_one_proposal:
                continue

            self.current_model = model
            self.current_state = state

            # Get the move-specific proposal.
            q, factors = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                branches_inds=inds_going_for_proposal,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # order everything properly
            q, _, new_branch_supps = self.ensure_ordering(
                list(state.branches.keys()), q, state.branches_inds, new_branch_supps
            )

            # if not wrapping with mutliple try (normal route)
            if not hasattr(self, "mt_ll") or not hasattr(self, "mt_lp"):
                # Compute prior of the proposed position
                logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

                self.fix_logp_gibbs(
                    branch_names_run, inds_run, logp, state.branches_inds
                )

                # Compute the lnprobs of the proposed position.
                # Can adjust supplementals in place
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=state.branches_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

            else:
                # if already computed in multiple try
                logl = self.mt_ll
                logp = self.mt_lp
                new_blobs = None

            # get log posterior
            logP = self.compute_log_posterior(logl, logp)

            # get previous information
            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # difference
            lnpdiff = factors + logP - prev_logP
            self.alpha = np.append(self.alpha, lnpdiff[0])

            # draw against acceptance fraction
            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # Update the parameters
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=new_blobs,
                inds=state.branches_inds,
                supplemental=new_supps,
                branch_supplemental=new_branch_supps,
            )
            state = self.update(state, new_state, accepted)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted
