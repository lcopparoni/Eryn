# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from eryn.moves.red_blue import RedBlueMove
import copy

__all__ = ["DEMove", "DESnookerMove"]


def ensure_sphere_boundary(costheta, phi):
    """This function makes sure that if theta is proposed outside of the boundary 
    we need to flip phi, the ranges are (theta in 0 pi), (phi in 0 2pi)"""
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    new_theta = np.arccos(z/np.sqrt(x*x + y*y+z*z))
    new_phi = np.sign(y)*np.arccos(x/np.sqrt(x*x + y*y))
    mask = (new_phi < 0.0)
    new_phi[mask] = new_phi[mask] + 2*np.pi
    return np.cos(new_theta), new_phi


def reflect_cosines_array(cos_ins,angle_ins,rotfac=np.pi,modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
        which requires also rotating the signal by rotfac each time, then mod the angle by modfac"""
    for itrk in range(cos_ins.size):
        if cos_ins[itrk] < -1.:
            cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
            angle_ins[itrk] += rotfac
        if cos_ins[itrk] > 1.:
            cos_ins[itrk] = 1.-(cos_ins[itrk]-1.)%4
            angle_ins[itrk] += rotfac
            #if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
            if cos_ins[itrk] < -1.:
                cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
                angle_ins[itrk] += rotfac
        angle_ins[itrk] = angle_ins[itrk]%modfac
    return cos_ins,angle_ins


class DEMove(RedBlueMove):

    """A Metropoplis step with a Differential Evolution Move based on the 
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
                sigma = 1e-5,
                factor = 100.0,
                modehopping = -1.0,
                method = "emcee",
                **kwargs
                ):
        names =  list(gamma_all.keys())
        self.gamma0 = dict()
        self.modehopping = dict()
        self.sigma = sigma
        self.log_factor = np.log(factor)
        if method == "emcee":
            self.get_factor = self._get_factor_emcee
        elif method == "bilby":
            self.get_factor = self._get_factor_bilby
        else:
            raise ValueError(f"method {method} is not available, only bilby and emcee are available")
        if isinstance(modehopping, float):
            self.modehopping = {name: modehopping for name in names}
            self.gamma0 = gamma_all
        else:
            for name in names:
            # Parse the proposal type.
                self.gamma0[name] = gamma_all[name]
                self.modehopping[name] = modehopping.get(name, -1.0)

      
        RedBlueMove.__init__(self, **kwargs) 




    def get_new_points(
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
        if mode_hop < self.modehopping[name]:
            #allow for mode hopping jump
            g0 = 1.0
        else:
            g0 = self.gamma0[name] * self.gamma_factor
        # get proper distance

        # Get the two complement points for each walker
        c1 = self.xp.take_along_axis(c, self.pairs[:, 0][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)
        c2 = self.xp.take_along_axis(c, self.pairs[:, 1][None, :, None, None], axis=1)  # (ntemps, Ns, nleaves_max, ndim)

        # Compute diff vectors
        if self.periodic is not None:
            diff = self.periodic.distance(
                {name: c1.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                {name: c2.reshape(ntemps * Ns, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, Ns, nleaves_max, ndim_here)
        else:
            diff = c2 - c1
        temp = s + g0 * diff        
        # wrap periodic values

        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
        return temp
    def _get_factor_emcee(self, ntemps, Ns, rng):
        return 1 + self.sigma * rng.randn(ntemps,Ns,1, 1)

    def _get_factor_bilby(self, ntemps, Ns, rng):
        scale0 = rng.randn(ntemps,Ns,1, 1)
        rescale = rng.uniform(-1.0,1.0, size = (ntemps,Ns,1, 1))* self.log_factor
        return scale0 * np.exp(rescale)
        # add a factorscale 

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate DE proposal

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
                pairs = _get_nondiagonal_pairs(Nc)
                indices = random_number_generator.choice(pairs.shape[0], size = Ns, replace = True)
                self.pairs = pairs[indices]
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # use DE to get new proposals
            newpos[name] = self.get_new_points(
                name, s, c, Ns, s.shape, i, random_number_generator
            )
        # proper factors
        factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()
        return newpos, factors

@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal."""
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return pairs

