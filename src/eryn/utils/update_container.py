import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import corner

# -*- coding: utf-8 -*-

ADAPTER_REGISTRY = {}


def register_adapter(*move_names):
    """Register an adapter class for one or more sampler move class names.

    Once decorated, ``Update_container`` discovers and drives the adapter
    automatically for any matching move in ``sampler.moves`` - no changes
    to ``Update_container`` are needed to wire up a new adapter.
    """

    def deco(cls):
        for name in move_names:
            ADAPTER_REGISTRY[name] = cls
        return cls

    return deco


@register_adapter("MALAMove")
class MALARescale(object):
    """Adaptive updater for :class:`MALAMove`, compatible with
    ``Update_container``'s per-move updater dispatch (same call
    convention as ``AdjustAMProposalScale`` / ``DESnookerRescale``:
    ``__call__(self, move, chain)``).

    Adapts the MALA step size ``epsilon`` towards ``target_acceptance``
    (0.574 is optimal for MALA, Roberts & Rosenthal 1998) using a
    Robbins-Monro schedule, and periodically refreshes the metric
    (mass matrix) from the current chain when the move uses a
    constant metric.
    """

    default_update_interval = 1 
    needs_chain = True

    def __init__(
        self,
        target_acceptance=0.574,
        window=100,
        cov_update=50,
        cov_update_scale=1.1,
        verbose=False,
    ):
        self.target_acceptance = target_acceptance
        self.window = window
        self.cov_update = cov_update
        self.cov_update_scale = cov_update_scale
        self.verbose = verbose

        self.time = 0
        self.last_cov_update = 0

    def __call__(self, move, chain=None):
        iteration = move.num_proposals

        # --- step size adaptation ---
        alpha = move.alpha
        if alpha.size > 0:
            window = min(self.window, alpha.size)
            alpha_cond = np.mean(np.exp(np.clip(alpha[-window:], -1e300, 0.0)))

            for name, eps in move.epsilon.items():
                delta_h = min(0.001 * eps, 1.0 / max(iteration, 1))
                if alpha_cond > self.target_acceptance:
                    move.epsilon[name] = eps + delta_h
                else:
                    move.epsilon[name] = eps - delta_h

                if self.verbose:
                    print(
                        f"[MALARescale] {name}: alpha={alpha_cond:.3f} "
                        f"eps={move.epsilon[name]:.5g}"
                    )

        # --- metric (mass matrix) refresh, only meaningful for constant_metric ---
        if (
            move.constant_metric
            and chain is not None
            and (iteration - self.last_cov_update) > self.cov_update
        ):
            for name in move.grad_function.keys():
                if name not in chain:
                    continue
                metric = np.cov(chain[name], rowvar=False)
                move.metric[name] = metric
                move.L[name] = cholesky((metric + metric.T) / 2, lower=True)

            self.last_cov_update = iteration
            self.cov_update *= self.cov_update_scale

        self.time += 1

    def get_state(self, move):
        return {"epsilon": dict(move.epsilon), "last_cov_update": self.last_cov_update}

    def set_state(self, move, state):
        move.epsilon.update(state["epsilon"])
        self.last_cov_update = state["last_cov_update"]


@register_adapter("FisherMove")
class FisherRescale(object):

    """Docstring for FisherRescale. """

    default_update_interval = 50
    needs_chain = False

    def __init__(
            self,
            target_acceptance = 0.234,
            fisher_matrix_function = None,
            recompute_fisher = False,
            verbose = False
            ):

        """Adjusted scale for Fisher proposal based on cold chain acceptance rate,
        if the method is provided  can also recompute the fisher

        """
        self.target_acceptance = target_acceptance
        self.verbose = verbose

        self.time = 0
        self.log_change = 0.0

        self.previously_accepted = 0
        self.previous_iter =  0
        """
        if fisher_matrix_function is None:
            self.recompute_fisher = False
        else:
            self.recompute_fisher = recompute_fisher
            self.fisher_matrix_function = fisher_matrix_function
        """
    def __call__(self, move, x0 = None):

        mean_af = 0.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (move.accepted[0] - self.previously_accepted)
                / (move.num_proposals - self.previous_iter)
                )

            if not np.isnan(mean_af):
                scale = 1./np.sqrt(self.time) * (mean_af - self.target_acceptance)
                self.log_change += scale
                for p in list(move.all_proposal.keys()):
                    move.all_proposal[p].svd *= np.exp(scale)
                    move.all_proposal[p].scale *= np.exp(2*scale)

            #TODO add recompute the fisher

        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1

    def get_state(self, move):
        return {"log_change": self.log_change}

    def set_state(self, move, state):
        log_scale = state["log_change"]
        for p in list(move.all_proposal.keys()):
            move.all_proposal[p].scale *= np.exp(2 * log_scale)
            move.all_proposal[p].svd *= np.exp(log_scale)
        self.log_change = log_scale


@register_adapter("DEMove")
class DERescale(object):

    default_update_interval = 50
    needs_chain = False

    def __init__(
            self,
            target_acceptance = 0.234,
            verbose=False,
            ):

        """Adjusted scale for Fisher proposal based on cold chain acceptance rate,
        if the method is provided  can also recompute the fisher

        """
        self.target_acceptance = target_acceptance
        self.verbose = verbose

        self.time = 0
        self.scale = 1.0
        self.previously_accepted = 0
        self.previous_iter =  0
    def __call__(self, move):

        mean_af = 0.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (move.accepted[0] - self.previously_accepted)
                / (move.num_proposals - self.previous_iter)
                )
            if np.isnan(mean_af):
                pass
            elif mean_af > 0.31:
                if self.scale < 2.0 :
                    for key in move.gamma0.keys():
                        self.scale *= 1.1
                        move.gamma0[key] *= 1.1
            elif mean_af < 0.2:
                if self.scale > 0.20:
                    for key in move.gamma0.keys():
                        self.scale *= 0.9
                        move.gamma0[key] *= 0.9
            else:
                for key in move.gamma0.keys():
                    self.scale *= np.sqrt(mean_af / self.target_acceptance)
                    move.gamma0[key] *= np.sqrt(mean_af / self.target_acceptance)
        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1

    def get_state(self, move):
        return {"scale": self.scale}

    def set_state(self, move, state):
        scale = state["scale"]
        for key in move.gamma0.keys():
            move.gamma0[key] *= scale
        self.scale = scale


@register_adapter("DESnookerMove")
class DESnookerRescale(object):

    default_update_interval = 50
    needs_chain = True

    def __init__(
            self,
            target_acceptance = 0.234,
            verbose=False,
            cov_function = dict(),# only updates when cov function is given
            update_scales = dict()

            ):

        """Adjusted scale for Fisher proposal based on cold chain acceptance rate,
        if the method is provided  can also recompute the fisher

        """
        self.target_acceptance = target_acceptance
        self.update_scales = update_scales
        self.cov_function = dict()
        for key in self.update_scales.keys():
            self.cov_function[key] = cov_function.get(key,np.cov)

        self.verbose = verbose
        self.scale = 1.0
        self.time = 0

        self.previously_accepted = 0
        self.previous_iter =  0
    def __call__(self, move, chain):

        mean_af = 0.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (move.accepted[0] - self.previously_accepted)
                / (move.num_proposals - self.previous_iter)
                )
            if np.isnan(mean_af):
                pass
            elif mean_af > 0.31:
                if self.scale < 2.0:
                    for key in move.gamma0.keys():
                        self.scale *= 1.1
                        move.gamma0[key] *= 1.1
            elif mean_af < 0.2:
                if self.scale > 0.20:
                    for key in move.gamma0.keys():
                        self.scale *= 0.9
                        move.gamma0[key] *= 0.9
            else:
                for key in move.gamma0.keys():
                    self.scale *= np.sqrt(mean_af / self.target_acceptance)
                    move.gamma0[key] *= np.sqrt(mean_af / self.target_acceptance)


            if self.time % 5 == 0:
                for key in self.cov_function.keys():
                    self.update_scales[key] = np.sqrt(np.diag(self.cov_function(chain[key], rowvar = False)))


        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1

    def get_state(self, move):
        return {"scale": self.scale}

    def set_state(self, move, state):
        scale = state["scale"]
        for key in move.gamma0.keys():
            move.gamma0[key] *= scale
        self.scale = scale


@register_adapter("SCAMMove")
class AdjustAMProposalScale(object):

    default_update_interval = 50
    needs_chain = True

    def __init__(
        self,
        cov_function = np.cov,
        target_acceptance=0.234,
        recompute_cov = True,
        verbose=False,
    ):
        """Adjusted scale for stretch proposal based on cold chain acceptance rate"""
        self.target_acceptance = target_acceptance
        self.verbose = verbose
        self.cov_function = cov_function
        self.recompute_cov = recompute_cov

        self.time = 0

        self.previously_accepted = 0
        self.previous_iter =  0
    def __call__(self, move, chain):

        mean_af = 0.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (move.accepted[0] - self.previously_accepted)
                / (move.num_proposals - self.previous_iter)
                )
            #TODO make this cleaner
            if np.isnan(mean_af):
                mean_af = 0.0
            if not np.isnan(mean_af):
                if self.recompute_cov:
                    if mean_af > self.target_acceptance:
                        for p in list(move.all_proposal.keys()):
                            cov_new = self.cov_function(chain[p], rowvar=False)
                            try:
                                U, S, V = np.linalg.svd(cov_new)
                                move.all_proposal[p].svd = (U, S, V)
                                move.all_proposal[p].scale = cov_new
                            except Exception as e:
                                print("WARNING: ", e, "unable to update covariance matrix for branch ", p )
                else:
                    scale = 1./np.sqrt(self.time) * (mean_af - self.target_acceptance)
                    for p in list(move.all_proposal.keys()):

                        U, S, v = move.all_proposal[p].svd
                        S *= np.exp(2*scale)
                        move.all_proposal[p].svd = (U, S,v)
                        move.all_proposal[p].scale *= np.exp(2*scale)


        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1

    def get_state(self, move):
        return {p: move.all_proposal[p].scale for p in move.all_proposal.keys()}

    def set_state(self, move, state):
        for p, cov in state.items():
            svd = np.linalg.svd(cov)
            move.all_proposal[p].scale = cov
            move.all_proposal[p].svd = svd


@register_adapter("StretchMove")
class Stretch_Update(object):

    default_update_interval = 100000
    needs_chain = False

    def __init__(
        self,
        target_acceptance=0.22,
        supression_factor=0.1,
        max_change=0.15,
        a_max = 3.0,
        a_min = 1.5,
        verbose=False,
    ):
        """
            Reimplementation of the default version in eryn

        Adjusted scale for stretch proposal based on cold chain acceptance rate"""
        self.target_acceptance = target_acceptance
        self.verbose = verbose
        self.max_change, self.supression_factor = max_change, supression_factor

        self.time = 0
        self.a_min = a_min
        self.a_max = a_max
        self.previously_accepted = 0
        self.previous_iter =  0
    def __call__(self, move):

        mean_af = 0.0
        change = 1.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (move.accepted[0] - self.previously_accepted)
                / (move.num_proposals - self.previous_iter)
            )

            if mean_af > self.target_acceptance:
                factor = self.supression_factor * (mean_af / self.target_acceptance)
                if factor > self.max_change:
                    factor = self.max_change
                change = 1 + self.supression_factor * factor

            else:
                factor = self.supression_factor * (self.target_acceptance / mean_af)
                if factor > self.max_change:
                    factor = self.max_change
                change = 1 - factor


            if np.isnan(change):
                pass
            else:
                new_a = move.a * change
                if new_a > self.a_min and new_a < self.a_max:
                    move.a = new_a


        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1

    def get_state(self, move):
        return {"a": move.a}

    def set_state(self, move, state):
        move.a = state["a"]


def _iter_moves(moves, prefix=()):
    """Recursively walk a sampler move list, descending into any move that
    exposes a ``.moves`` attribute (e.g. ``CombineMove``), so adapter
    discovery does not need to special-case container move types by name.

    Yields ``(inds, move)`` where ``inds`` is an int for a top-level move
    or a tuple of ints for a move nested inside container move(s).
    """
    for i, mv in enumerate(moves):
        inds = prefix + (i,)
        sub_moves = getattr(mv, "moves", None)
        if sub_moves is not None:
            yield from _iter_moves(sub_moves, inds)
        else:
            yield (inds[0] if len(inds) == 1 else inds), mv


def _resolve_move(sampler_moves, inds):
    """Inverse of ``_iter_moves``: fetch the move object for an ``inds`` key."""
    if isinstance(inds, tuple):
        mv = sampler_moves[inds[0]]
        for j in inds[1:]:
            mv = mv.moves[j]
        return mv
    return sampler_moves[inds]


class Update_container(object):

    """Sampler update wrapper.

    Drives every registered adapter (see ``register_adapter``) found among
    ``sampler.moves`` without needing to know about specific adapter or move
    types - adding a new adapter class elsewhere in this module is enough
    for it to be picked up here automatically.
    """

    def __init__(self,
                 sampler,
                 updater_kwargs = None,
                 update_intervals = None,
                 plot_iteration = 250,
                 backend_dir  = None,
                 acc_rate_step = 25,
                 plot_kwargs  = None,
                 stop_update = int(3e4),
                 dump_interval = 50,
                 ):
        """
        :sampler: the eryn sampler this container will be called on
        :updater_kwargs: dict of {move_class_name: kwargs} passed to that
            move's registered adapter constructor
        :update_intervals: dict of {move_class_name: interval} overriding
            the adapter's ``default_update_interval``
        :plot_iteration: outputs a corner plot every x steps
        :acc_rate_step: how often to log the acceptance rate
        :dump_interval: how often to dump adapter state to disk
        :stop_update: end update procedure after given steps
        """
        self._plot_iteration = plot_iteration
        self._acc_rate_step = acc_rate_step
        self._stop_update = stop_update
        self._dump_interval = dump_interval
        self.plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
        updater_kwargs = updater_kwargs if updater_kwargs is not None else dict()
        update_intervals = update_intervals if update_intervals is not None else dict()

        if backend_dir is None:
            print("HEY NO MOVE BACKEND, setting it as move_backend")
            self.backend_dir = "MCMC_raw"
        else:
            self.backend_dir = backend_dir

        # discover and instantiate an adapter for every move with a
        # registered adapter class, at any nesting depth
        self.updaters = {}
        self.update_intervals = {}
        for inds, mv in _iter_moves(sampler.moves):
            move_name = mv.__class__.__name__
            adapter_cls = ADAPTER_REGISTRY.get(move_name)
            if adapter_cls is None:
                continue
            self.updaters[inds] = adapter_cls(**updater_kwargs.get(move_name, {}))
            self.update_intervals[inds] = update_intervals.get(
                move_name, adapter_cls.default_update_interval
            )

    def _dump_move_info(self, sampler):
        output = {}
        for inds, updater in self.updaters.items():
            move = _resolve_move(sampler.moves, inds)
            output[inds] = updater.get_state(move)
        np.save(self.backend_dir + "/moves", output)

    def _load_move_info(self, sampler):
        output = np.load(self.backend_dir + "/moves.npy", allow_pickle=True).item()
        for inds, updater in self.updaters.items():
            if inds not in output:
                continue
            move = _resolve_move(sampler.moves, inds)
            updater.set_state(move, output[inds])

    @staticmethod
    def _build_chain(samp, discard):
        samples = samp.get_chain(discard=discard)
        return {
            p: samples[p][:, 0, :, 0].reshape(-1, samp.ndims[p])
            for p in samp.branch_names
        }

    def _write_acc_rate(self, samp, current_it):
        acc_rates_txt = [f"{np.mean(mv.acceptance_fraction[0]):.5f}" for mv in samp.moves]
        swaps_rate = "\t".join([f"{x:.5f}" for x in samp.swap_acceptance_fraction[:3]])
        with open(self.backend_dir + "/acc_rate.txt", "a") as f0:
            f0.write(f"{current_it}\t" + "\t".join(acc_rates_txt) + "\t" + swaps_rate + "\n")

    def _make_corner_plots(self, samp, discard):
        samples = samp.get_chain(discard=discard)
        logl = samp.get_log_like(discard=discard)[:, 0].flatten()
        for p in samp.branch_names:
            chain = np.hstack((samples[p][:, 0].reshape(-1, samp.ndims[p]), logl[:, None]))
            corner_kwargs = self.plot_kwargs.get(p, dict())
            fig = corner.corner(
                    chain,
                     bins = 30,
                     plot_datapoints = False,
                     levels = [0.39346934, 0.86466472, 0.988891  ],
                     label_kwargs = {"size": 28},
                     **corner_kwargs
                     )
            fig.suptitle(self.backend_dir + f"/corner_branch_{p}.png", size = 25)

            fig.savefig(self.backend_dir + f"/corner_branch_{p}.png", bbox_inches = "tight")
            plt.close()

    def __call__(self, i, last, samp):
        current_it = samp.iteration
        discard = int(samp.iteration * 0.8)

        # Print the current acceptance rate
        if (current_it > self._acc_rate_step) and (current_it % self._acc_rate_step == 0):
            self._write_acc_rate(samp, current_it)

        if current_it < self._stop_update:
            due_updaters = [
                (inds, updater)
                for inds, updater in self.updaters.items()
                if current_it > self.update_intervals[inds]
                and current_it % self.update_intervals[inds] == 0
            ]

            chain = None
            if any(updater.needs_chain for _, updater in due_updaters):
                chain = self._build_chain(samp, discard)

            for inds, updater in due_updaters:
                move = _resolve_move(samp.moves, inds)
                if updater.needs_chain:
                    updater(move, chain)
                else:
                    updater(move)

        if (current_it > 1) and (current_it % self._dump_interval == 0):
            self._dump_move_info(samp)

        # plot the corner
        if (current_it % self._plot_iteration == 0) and (current_it > 0):
            self._make_corner_plots(samp, discard)
