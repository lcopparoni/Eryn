# -*- coding: utf-8 -*-

import numpy as np

from eryn.moves.mh import MHMove
import copy

__all__ = ["SCAMMove"]


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


class SCAMMove(MHMove):
    """Adaptive Metropolis / Single Component Adaptive Metropolis (AM/SCAM) move.

    The proposal is made in the eigenbasis of the covariance matrix
    ``cov = U diag(S) U^T``. With probability 1/2 all the eigen-directions are
    updated at once (AM), otherwise only one randomly chosen eigen-direction
    is updated (SCAM). The jump along eigen-direction ``j`` is

    .. math::

        \\delta y_j = f \\, \\sqrt{S_j} \\, \\frac{2.38}{\\sqrt{n_{dim} \\beta}} \\, z,
        \\quad z \\sim \\mathcal{N}(0, 1),

    where ``f`` is the optional random ``factor`` and ``beta`` the inverse
    temperature (if ``scale_temperature`` is ``True``). The proposal is
    symmetric, so the proposal factors are zero.

    Args:
        cov_all (dict): Keys are branch names. Values are the proposal covariance,
            given either as a ``(ndim,)`` vector (diagonal covariance) or as a
            ``(ndim, ndim)`` matrix.
        factor (float, optional): If provided, the proposal scale is multiplied
            by ``exp(U(-log(factor), log(factor)))``. Must be ``>= 1``.
            (default: ``None``)
        indx_list (list, optional): List of ``(branch_name, mask)`` tuples.
            ``mask`` is a boolean array of shape ``(1, ndim)`` marking a block
            of parameters. For each walker one block is chosen at random and
            only those parameters are updated. (default: ``None``)
        sky_periodic (list, optional): List of ``(branch_name, inds)`` tuples,
            where ``inds`` holds the indices of ``(cos(theta), phi)`` sky
            parameters to reflect around the poles. (default: ``None``)
        abs_value (int or array-like, optional): Indices of parameters whose sign
            is flipped at random, for parameters entering only through their
            absolute value. (default: ``None``)
        scale_temperature (bool, optional): If ``True``, scale the jump by
            ``1 / sqrt(beta)`` of each chain. A chain with ``beta = 0`` uses the
            previous chain's ``beta``. (default: ``False``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Attributes:
        cov (dict): Current proposal covariance for each branch.
        svd (dict): ``(U, S, V)`` decomposition of ``cov`` for each branch.

    Raises:
        ValueError: If the covariance dimensions are invalid or ``factor < 1``.

    """

    def __init__(
        self,
        cov_all,
        factor=None,
        indx_list=None,
        sky_periodic=None,
        abs_value=None,
        scale_temperature=False,
        **kwargs,
    ):

        self.cov = {}
        self.svd = {}
        for name, cov in cov_all.items():
            cov = np.atleast_1d(cov)
            if len(cov.shape) == 1:
                # A diagonal proposal was given.
                cov = np.diag(cov)
            elif not (len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]):
                raise ValueError("Invalid proposal scale dimensions")

            self.cov[name] = cov
            self.svd[name] = np.linalg.svd(cov)

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        # propose in blocks
        self.indx_list = indx_list
        # ensure sky periodicity
        self.sky_periodic = sky_periodic
        # absolute value variable
        self.abs_value = abs_value
        self.scale_temperature = scale_temperature
        super(SCAMMove, self).__init__(**kwargs)

    def get_factor(self, rng):
        """Random scale factor ``exp(U(-log(factor), log(factor)))``."""
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, name, rng, x0, betas):
        """Make an AM/SCAM jump.

        Args:
            name (str): Branch name.
            rng (object): Random state object.
            x0 (np.ndarray): Current points, shape ``(n, ndim)``.
            betas (np.ndarray): Inverse temperatures, shape ``(n,)``.

        Returns:
            np.ndarray: Proposed points, shape ``(n, ndim)``.

        """
        U, S, _ = self.svd[name]
        nw, nd = x0.shape

        # go in eigen basis
        y = x0 @ U

        scale = self.get_factor(rng)
        if rng.random() > 0.5:
            # move along only one uncorrelated direction SCAM
            rand_j = rng.permutation(nd)[:1]
        else:
            # move along all of them AM
            rand_j = np.arange(nd)

        y[:, rand_j] += (
            scale * rng.normal(size=nw)[:, None] * np.sqrt(S[None, rand_j])
            * 2.38 / np.sqrt(nd * betas[:, None])
        )

        # go back to the original basis
        return y @ U.T

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get AM/SCAM proposal

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)

        """

        # initialize ouput
        q = {}
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]
            inds_here = np.where(inds == True)

            betas = np.ones((ntemps, nwalkers, nleaves_max))

            if self.scale_temperature and self.temperature_control is not None:
                betas_tmp = copy.copy(self.temperature_control.betas)
                if betas_tmp[-1] == 0.0:
                    betas_tmp[-1] = betas_tmp[-2]
                betas = betas * betas_tmp[:, None, None]
            betas_calc = betas[inds_here]

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords_tmp = self.get_updated_vector(
                name, random, coords[inds_here], betas_calc
            )

            # update only a block of parameters, this helps for the search phase
            if self.indx_list is not None:
                new_coords = coords[inds_here].copy()
                indx_list_here = np.asarray([el[1] for el in self.indx_list if el[0]==name])
                nw = new_coords_tmp.shape[0]
                # list of numbers indicating wich group of parameters to change
                ind_to_chage = random.randint(len(indx_list_here), size=nw)
                mask = indx_list_here[ind_to_chage][:, 0, :]
                new_coords[mask] = new_coords_tmp[mask]
            else:
                new_coords = new_coords_tmp

            # enforce positive proposal
            if self.abs_value is not None:
                # change the sign of the parameter randomly
                sign_par = random.choice([-1.0, 1.0], size=new_coords[..., self.abs_value].shape)
                new_coords[..., self.abs_value] *= sign_par

            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                for el in indx_list_here:
                    sky_inds = el[0]
                    csth = new_coords[:, sky_inds][:, 0]
                    ph = new_coords[:, sky_inds][:, 1]
                    new_coords[:, sky_inds] = np.asarray(reflect_cosines_array(csth, ph)).T

            # put into coords in proper location
            q[name][inds_here] = new_coords

        # handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                q[name] = self.periodic.wrap(
                        {name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)}
                        )[name].reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros((ntemps, nwalkers))
