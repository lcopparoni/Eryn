# -*- coding: utf-8 -*-

import numpy as np

from eryn.moves.mh import MHMove
import copy

__all__ = ["SCAMMove", "FisherMove"]


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
    """A Metropolis step with a Gaussian proposal function.

    This class is heavily based on the same class in ``emcee``. 

    Args:
        cov (dict): The covariance of the proposal function. The keys are branch names and the 
            values are covariance information. This information can be provided as a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general, respectively.
        mode (str, optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn. (default: ``"vector"``)
        factor (float, optional): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode. (default: ``None``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(self, cov_all, mode="AM", factor=None, indx_list=None, sky_periodic=None, abs_value=None, prop=None, scale_temperature = False, **kwargs):

        self.all_proposal = {}
        self.list_prop = []
        for name, cov in cov_all.items():
            # Parse the proposal type.
            try:
                float(cov)

            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    # A diagonal proposal was given.
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, "vector")

                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                    ndim = cov.shape[0]
                    list_mode = mode.split(",")
                    
                    if len(list_mode)>1:
                        
                        # function that propose randomly from AM and DE, make a list where you append proposal
                        for el in list_mode:

                            if el=="Gaussian":
                                proposal = _proposal(cov, factor,"vector")
                            if el=="AM":
                                proposal = AM_proposal(cov, factor, "vector")
                            if el=="Fisher":
                                proposal = Fisher_proposal(cov, factor, "vector")
                            self.list_prop.append(proposal)
                    else:
                        if mode=="Gaussian":
                            proposal = _proposal(cov, factor, "vector")
                        if mode=="AM":
                            proposal = AM_proposal(cov, factor, "vector")
                        if mode=="Fisher":
                            proposal = Fisher_proposal(cov, factor, "vector")
                        self.list_prop.append(proposal)
                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor,  "vector")
            self.all_proposal[name] = proposal

        # propose in blocks
        self.indx_list = indx_list
        # ensure sky periodicity
        self.sky_periodic = sky_periodic
        # absolute value variable
        self.abs_value = abs_value
        self.scale_temperature = scale_temperature
        super(SCAMMove, self).__init__(**kwargs)
        # the definition of the temperature scaled fisher requires the temperature control to be defined



    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution

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
            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            betas = np.ones((ntemps, nwalkers, nleaves_max))
            
            if self.scale_temperature and  self.temperature_control is not None:
                betas_tmp = copy.copy(self.temperature_control.betas)
                if betas_tmp[-1] == 0.0:
                    betas_tmp[-1] = betas_tmp[-2]
                #breakpoint()
                betas = betas *betas_tmp
            betas_calc = betas[inds_here]

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords_tmp = coords[inds_here].copy()
            new_coords = coords[inds_here].copy()
            # random choice from list of proposal
            proposal_fn = np.random.choice(self.list_prop, size=1)
            new_coords_tmp = proposal_fn[0](coords[inds_here], random, betas_calc)[0]
            
            # swap walkers, this helps for the search phase    
            if self.indx_list is not None:
                indx_list_here = np.asarray([el[1] for el in self.indx_list if el[0]==name])
                nw = new_coords_tmp.shape[0]
                # list of numbers indicating wich group of parameters to change
                ind_to_chage = np.random.randint(len(indx_list_here),size=nw)
                new_coords[indx_list_here[ind_to_chage][:,0,:]] = new_coords_tmp[indx_list_here[ind_to_chage][:,0,:]]
            else:
                new_coords = new_coords_tmp.copy()
            
            # enforce positive proposal
            if self.abs_value is not None:
                # change the sign of the parameter randomly
                sign_par = np.random.choice([-1.0,1.0],size=new_coords[...,self.abs_value].shape)
                new_coords[...,self.abs_value] *= sign_par

            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                nw = new_coords_tmp.shape[0]
                for temp_ind in range(len(indx_list_here)):
                    csth = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,0]
                    ph = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,1]
                    new_coords[:,indx_list_here[temp_ind][0]] = np.asarray(reflect_cosines_array(csth, ph)).T
                

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                q[name] = self.periodic.wrap({name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)})
                q[name] = tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros((ntemps, nwalkers))


class _isotropic_proposal(object):

    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode, prop=None):
        self.index = 0
        self.scale = scale
        self.svd = None
        self.chain = None
        self.mean = None
        self.loglambda = 1.0
        self.invscale = np.linalg.inv(np.linalg.cholesky(scale))
        self.use_current_state = True
        self.crossover = False
        self.propose_transform = prop
        
        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                ("'{0}' is not a recognized mode. " "Please select from: {1}").format(
                    mode, self.allowed_modes
                )
            )
        self.mode = mode
        
    def update_proposal(self, new_X, gamma, delta_alpha):
        if self.mean is None:
            self.mean = np.zeros(self.scale.shape[0])
        self.loglambda += gamma * delta_alpha
        self.mean += gamma * (new_X - self.mean)
        self.scale += gamma * ((new_X - self.mean) @ (new_X - self.mean).T - self.scale)
        self.scale *= self.loglambda

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0, betas):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape)) / np.sqrt(betas[:,None])

    def __call__(self, x0, rng, betas):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0, betas)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):
    def get_updated_vector(self, rng, x0, betas):
        scale = self.scale * betas
        return x0 + self.get_factor(rng) * scale * rng.randn(*(x0.shape)) / np.sqrt(betas[:, None])


class _proposal(_isotropic_proposal):

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0, betas):
        y = rng.multivariate_normal(np.zeros(len(self.scale)), self.scale, size=len(x0))
        return x0 + self.get_factor(rng)  * y / np.sqrt(betas[:, None])



class AM_proposal(_isotropic_proposal):
    """
    Adaptive Jump Proposal.
    Single Component Adaptive Jump Proposal.
    """
    allowed_modes = ["vector"]
    
    def get_updated_vector(self, rng, x0, betas):
        if self.svd is None:
            svd = np.linalg.svd(self.scale)
        else:
            svd = self.svd

        new_pos = x0.copy()
        nw, nd = new_pos.shape
        U, S, v = svd

        # adjust step size
        prob = rng.random()
    
        # go in eigen basis
        y = np.dot(U.T,x0.T).T # np.asarray([np.dot(U.T, x0[i]) for i in range(nw)])
        # choose a random parameter in the uncorrelated basis
        ind_vec = np.arange(nd)
        scale = self.get_factor(rng)
        if prob>0.5:
            # move along only one uncorrelated direction SCAM
            np.random.shuffle(ind_vec)
            rand_j = ind_vec[:1]
        else:
            # move along all of them AM
            rand_j = ind_vec
    
        y[:,rand_j] += scale * np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(nd * betas[:, None])
    
        # go back to the basis
        new_pos = np.dot(U,y.T).T # np.asarray([np.dot(U, y[i]) for i in range(nw)]) 

        return new_pos


class FisherMove(MHMove):
    """A Metropolis step with a Fisher Gaussian proposal function.

    This class is heavily based on the same class in ``emcee``. 

    Args:
        cov (dict): The covariance of the proposal function. The keys are branch names and the 
            values are covariance information. This information can be provided as a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general, respectively.
        mode (str, optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn. (default: ``"vector"``)
        factor (float, optional): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode. (default: ``None``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(self, cov_all, factor=None, indx_list=None, sky_periodic=None, abs_value=None, scale_temperature = False, **kwargs):

        self.all_proposal = {}
        self.list_prop = []
        for name, cov in cov_all.items():
            # Parse the proposal type.
            if  len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                ndim = cov.shape[0]
                proposal = Fisher_proposal(cov, factor, "vector")
                self.list_prop.append(proposal)
                self.all_proposal[name] = proposal
            else:
                raise ValueError("Invalid proposal scale dimensions")
        # propose in blocks
        self.indx_list = indx_list
        # ensure sky periodicity
        self.sky_periodic = sky_periodic
        # absolute value variable
        self.abs_value = abs_value
        self.scale_temperature = scale_temperature
        super(FisherMove, self).__init__(**kwargs)
        # the definition of the temperature scaled fisher requires the temperature control to be defined



    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution

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
            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            betas = np.ones((ntemps, nwalkers, nleaves_max))
            
            if self.scale_temperature and  self.temperature_control is not None:
                betas_tmp = copy.copy(self.temperature_control.betas)
                if betas_tmp[-1] == 0.0:
                    betas_tmp[-1] = betas_tmp[-2]
                #breakpoint()
                betas = betas *betas_tmp
            betas_calc = betas[inds_here]

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords_tmp = coords[inds_here].copy()
            new_coords = coords[inds_here].copy()
            # random choice from list of proposal
            proposal_fn = np.random.choice(self.list_prop, size=1)
            new_coords_tmp = proposal_fn[0](coords[inds_here], random, betas_calc)[0]
            
            # swap walkers, this helps for the search phase
            if self.indx_list is not None:
                indx_list_here = np.asarray([el[1] for el in self.indx_list if el[0]==name])
                nw = new_coords_tmp.shape[0]
                # list of numbers indicating wich group of parameters to change
                ind_to_chage = np.random.randint(len(indx_list_here),size=nw)
                new_coords[indx_list_here[ind_to_chage][:,0,:]] = new_coords_tmp[indx_list_here[ind_to_chage][:,0,:]]
            else:
                new_coords = new_coords_tmp.copy()
            
            # enforce positive proposal
            if self.abs_value is not None:
                # change the sign of the parameter randomly
                sign_par = np.random.choice([-1.0,1.0],size=new_coords[...,self.abs_value].shape)
                new_coords[...,self.abs_value] *= sign_par

            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                nw = new_coords_tmp.shape[0]
                for temp_ind in range(len(indx_list_here)):
                    csth = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,0]
                    ph = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,1]
                    new_coords[:,indx_list_here[temp_ind][0]] = np.asarray(reflect_cosines_array(csth, ph)).T
                

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                q[name] = self.periodic.wrap({name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)})
                q[name] = tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros((ntemps, nwalkers))




class Fisher_proposal(_isotropic_proposal):
    """
    Fisher Jump Proposal.
    It randomly jumps either in all directions or along one of the uncorreleated directions
    Single Component Adaptive Jump Proposal.
    """
    allowed_modes = ["vector"]
    
    def get_updated_vector(self, rng, x0, betas):
        if self.svd is None:
            svd = np.linalg.svd(self.scale)
        else:
            svd = self.svd

        new_pos = x0.copy()
        nw, nd = new_pos.shape
        U, S, v = svd

        # adjust step size
        prob = rng.random()
    
        scale = self.get_factor(rng)
        if prob>0.5:
            # move along only one of the uncorrelated direction 
            y = np.dot(U.T,x0.T).T # np.asarray([np.dot(U.T, x0[i]) for i in range(nw)])
            # choose a random parameter in the uncorrelated basis
            ind_vec = np.arange(nd)
            # move along only one uncorrelated direction SCAM
            np.random.shuffle(ind_vec)
            rand_j = ind_vec[:1]

            y[:,rand_j] += scale * np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(nd * betas[:, None])

            new_pos = np.dot(U,y.T).T # np.asarray([np.dot(U, y[i]) for i in range(nw)]) 
        else:
            # Move along all the direction 
            new_pos +=  scale * np.random.multivariate_normal(np.zeros(nd), self.scale,size=nw) * np.sqrt(S[None,:]) * 2.38 / np.sqrt(nd * betas[:, None])

        return new_pos
