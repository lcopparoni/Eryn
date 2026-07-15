# -*- coding: utf-8 -*-
import numpy as np
from itertools import permutations

from eryn.moves.red_blue import RedBlueMove
import copy

__all__ = ["DESnookerMove"]



def _get_noindiagonal_triplets(n: int) -> np.ndarray:
    """All (i, j, k) where i, j, k in [0, n) and all different."""
    idx = np.array(list(permutations(range(n), 3)))
    return idx

class DESnookerMove(RedBlueMove):

    """A RedBlueMove step with a Differential Evolution Snooker Move on the 
    Emcee code
    or bilby code 

    Args:
        gamma (dict): scale for the DE component
        sigma0 (float) jittering of the gammas
        modehopping (float=0.0) : probability of modehopping jump (gamma = 1.0)

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.



    """

    def __init__(
                self,
                gamma_all = dict(),
                scale_all = dict(),
                sigma = 1e-5,
                factor = 100.0,
                method = "emcee",
                **kwargs
                ):
        names =  list(gamma_all.keys())
        self.gamma0 = dict()
        self.scale = dict()
        self.modehopping = dict()
        self.sigma = sigma
        self.log_factor = np.log(factor)
        if method == "emcee":
            self.get_factor = self._get_factor_emcee
        elif method == "bilby":
            self.get_factor = self._get_factor_bilby
        else:
            raise ValueError(f"method {method} is not available, only bilby and emcee are available")
        for name in names:
            # Parse the proposal type.
            self.gamma0[name] = gamma_all[name]
            self.scale[name] = scale_all.get(name, 1)
       
        #kwargs["n_splits"]  = kwargs.get("n_splits", 4)
      
        RedBlueMove.__init__(self, **kwargs) 


    def _get_factor_emcee(self, ntemps, Ns, rng):
        return 1 + self.sigma * rng.randn(ntemps,Ns,1, 1)

    def _get_factor_bilby(self, ntemps, Ns, rng):
        scale0 = rng.randn(ntemps,Ns,1, 1)
        rescale = rng.uniform(-1.0,1.0, size = (ntemps,Ns,1, 1))* self.log_factor
        return scale0 * np.exp(rescale)
        # add a factorscale 

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate DE snooker proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """

        # needs to be set before we reach the end
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}
        new_factor = {}
        # iterate over branches

        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            # get compliment possibilities
            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            # need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns
                self.gamma_factor = self.get_factor(ntemps,Ns, random_number_generator)
                # Get the pair indices
                triplets = _get_noindiagonal_triplets(Nc)
                indices = random_number_generator.choice(triplets.shape[0], size = Ns, replace = True)
                self.triplets = triplets[indices]
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # use DE to get new proposals
            newpos[name], new_factor[name] = self.get_new_points_factors(
                name, s, c, Ns, s.shape, i, random_number_generator
            )
        # proper factors
        #factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        factors = sum(new_factor.values())
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()
        return newpos, factors

    

    def get_new_points_factors(
        self, name, s, c, Ns, branch_shape, branch_i, random_number_generator
    ):
        """Get mew points in DE move.

        Takes compliment and uses it to get new points for those being proposed.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c (np.ndarray): Compliment to move points with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            Ns (int): Number to generate.
            branch_shape (tuple): Full branch shape.
            branch_i (int): Which branch in the order is being run now. This ensures that the
                randomly generated quantity per walker remains the same over branches.
            random_number_generator (object): Random state object.

        Returns:
            np.ndarray: New proposed points with shape ``(ntemps, Ns, nleaves_max, ndim)``.


        """
        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape
        mode_hop = random_number_generator.random()
        g0 = self.gamma0[name] * self.gamma_factor
        # get proper distance

        # Get the two complement points for each walker
        c0 = self.xp.take_along_axis(c, self.triplets[:, 0][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)
        c1 = self.xp.take_along_axis(c, self.triplets[:, 1][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)
        c2 = self.xp.take_along_axis(c, self.triplets[:, 2][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)
        if self.periodic is not None:
            delta = self.periodic.distance(
            {name: s.reshape(ntemps * Ns, nleaves_max, ndim_here)},
            {name: c0.reshape(ntemps * Ns, nleaves_max, ndim_here)},
            xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here) / self.scale[name]
        else:
            delta = (s - c0) / self.scale[name]
        
        norm = np.sqrt(np.sum(delta* delta, axis = -1))
        delta /= norm[...,None]
        z1 = np.sum(delta* c1/self.scale[name], axis = -1)
        z2 = np.sum(delta* c2/self.scale[name], axis = -1)

        temp = s + g0 * delta * self.scale[name] * (z1 - z2)[:,:,:,None]
        # wrap periodic values

        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)
            new_delta = self.periodic.distance(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                {name: c0.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here) /self.scale[name]
        else:
            new_delta = (temp - c0) / self.scale[name]
        new_norm = np.sqrt(np.sum(new_delta*new_delta,axis = -1))
        factor = np.sum((ndim_here-1.0)*(np.log(new_norm) - np.log(norm)),axis =-1)
        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
            factor = factor.get()
        #breakpoint()
        return temp, factor



