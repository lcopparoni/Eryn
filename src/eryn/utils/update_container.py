import numpy as np
import matplotlib.pyplot as plt
import corner


# TODO
# - Move load and dump inside the move updater definition
# - Create generic updater class

class FisherRescale(object):

    """Docstring for FisherRescale. """

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

class DERescale(object):
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

class DESnookerRescale(object):
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



class AdjustAMProposalScale(object):
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
                                svd = np.linalg.svd(cov_new)
                                move.all_proposal[p].svd = svd
                                move.all_proposal[p].scale = cov_new
                            except Exception as e:
                                print("WARNING: ", e, "unable to update covariance matrix for branch ", p )
                else:
                    scale = 1./np.sqrt(self.time) * (mean_af - self.target_acceptance)
                    for p in list(move.all_proposal.keys()):
                        move.all_proposal[p].svd *= np.exp(scale) 
                        move.all_proposal[p].scale *= np.exp(2*scale)


                        
                
        self.previously_accepted = move.accepted[0].copy()
        self.previous_iter = move.num_proposals
        self.time += 1


class Stretch_Update(object):
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
# TODO
# Rethink this in a nicer way

class Update_container(object):

    """sampler update wrapper"""

    def __init__(self,
                 sampler,
                 stretch_update = 100000,
                 AM_update = 50,
                 DE_update = 50,
                 Fisher_update = 50,
                 Snooker_update = 50,
                 Stretch_kwargs = dict(),
                 AM_kwargs = dict(),
                 DE_kwargs = dict(),
                 Fisher_kwargs = dict(),
                 Snooker_kwargs = dict(),
                 plot_iteration = 250,
                 backend_dir  = None,
                 acc_rete_output = None,
                 acc_rate_step = 25,
                 plot_kwargs  = dict(),
                 stop_update = int(3e4)
                 ):
        """TODO: to be defined.

        :stretch_update: number of steps after which stretch is updated
        :AM_update: number of steps after which stretch is updated
        :DE_update: number of steps after which DE is updated
        :Fisher_update: number of steps after which Fisher is updated
        :acc_rete_output: if not None prints the acc rate on a given acc_rate_file
        : plot_iteration: outputs a plot every x step
        : stop_update: End update procedure after given steps
        :): TODO

        """
        self._stretch_update = stretch_update
        self._AM_update = AM_update
        self._DE_update = DE_update
        self._Fisher_update = Fisher_update
        self._Snooker_update = Snooker_update
        self._plot_iteration = plot_iteration 
        self._acc_rate_step = acc_rate_step
        self._stop_update = stop_update

        self.Fisher_updaters  = {}
        self.Stretch_updaters = {}
        self.AM_updaters      = {}
        self.DE_updaters      = {}
        self.Snooker_updaters = {}
        self.plot_kwargs      = plot_kwargs

        if backend_dir is None:
            print("HEY NO MOVE BACKEND, setting it as move_backend")
            self.backend_dir = "MCMC_raw"
        else:
            self.backend_dir = backend_dir
    
        # set the moves by move kind
        move_names = [p.split("_")[0] for p in sampler.move_keys]
        for i, m in enumerate(move_names):
            if m == "StretchMove":
                self.Stretch_updaters.update({i: Stretch_Update(**Stretch_kwargs)})
            elif m == "SCAMMove":
                self.AM_updaters.update({i: AdjustAMProposalScale(**AM_kwargs)})
            elif m == "FisherMove":
                self.Fisher_updaters.update({i: FisherRescale(**Fisher_kwargs)})
            elif m == "DEMove":
                self.DE_updaters.update({i: DERescale(**DE_kwargs)})
            elif m == "DESnookerMove":
                self.Snooker_updaters.update({i: DESnookerRescale(**Snooker_kwargs)})
            elif m == "CombineMove":
                # subset update for the combine moves

                for j, mv in enumerate(sampler.moves[i].moves):
                    m_name = mv.__class__.__name__

                    if m_name == "StretchMove":
                        self.Stretch_updaters.update({(i,j): Stretch_Update(**Stretch_kwargs)})
                    elif m_name == "SCAMMove":
                        self.AM_updaters.update({(i,j): AdjustAMProposalScale(**AM_kwargs)})
                    elif m_name == "FisherMove":
                        self.Fisher_updaters.update({(i,j): FisherRescale(**Fisher_kwargs)})
                    elif m_name == "DEMove":
                        self.DE_updaters.update({(i,j): DERescale(**DE_kwargs)})
        """
        # Currently implemented so that each move has the same structure
        self.AM_updater = [AdjustAMProposalScale(**AM_kwargs) for _ in self._AM_indices]
        self.stretch_updater = [Stretch_Update(**Stretch_kwargs) for _ in self._stretch_indices]
        # TODO define this parameters better 
        #self.DE_updater = [DE_Update(**Stretch_kwargs) for _ in self._DE_indices]
        #self.Fisher_updater = [Fisher_Update(**Stretch_kwargs) for _ in self._Fisher_indices]
        self.combine_updater = {}
        """
    def _dump_move_info(self, sampler):
        output = {
            "AM": {},
            "Stretch": {},
            "DE": {},
            "Fisher": {},
            "Snooker": {}
                }
        for inds, updater in self.AM_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                branches = list(sampler.moves[i].moves[j].all_proposal.keys())
                output["AM"].update({(i,j): {p: sampler.moves[i].moves[j].all_proposal[p].scale for p in branches}})
            else:
                branches = list(sampler.moves[inds].all_proposal.keys())
                output["AM"].update({inds: {p: sampler.moves[inds].all_proposal[p].scale for p in branches}})

        for inds, updater in self.Stretch_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                output["Stretch"].update({(i,j): sampler.moves[i].moves[j].a})
            else:
                output["Stretch"].update({inds: sampler.moves[inds].a})
        for inds, updater in self.DE_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                output["DE"].update({(i,j): updater.scale})
            else:
                output["DE"].update({inds: updater.scale})
        for inds, updater in self.Fisher_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                output["Fisher"].update({(i,j): updater.log_change})
            else:
                output["Fisher"].update({inds: updater.log_change})

        for inds, updater in self.Snooker_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                output["Snooker"].update({(i,j): updater.scale})
            else:
                output["Snooker"].update({inds: updater.scale})
        np.save(
                self.backend_dir + "/moves",
                output)
    def _load_move_info(self, sampler):
        output = np.load(self.backend_dir + "/moves.npy", allow_pickle=True).item()
        for inds, updater in self.AM_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                for p in  list(sampler.moves[i].moves[j].all_proposal.keys()):
                    cov = output["AM"][(i,j)][p]
                    svd = np.linalg.svd(cov)
                    sampler.moves[i].moves[j].all_proposal[p].scale = cov
                    sampler.moves[i].moves[j].all_proposal[p].svd = svd

            else:
                for p in  list(sampler.moves[inds].all_proposal.keys()):
                    cov = output["AM"][inds][p]
                    svd = np.linalg.svd(cov)
                    sampler.moves[inds].all_proposal[p].scale = cov
                    sampler.moves[inds].all_proposal[p].svd = svd


        for inds, updater in self.Stretch_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                sampler.moves[i].moves[j].a = output["Stretch"][(i,j)]
            else:
                sampler.moves[inds].a = output["Stretch"][inds]
        for inds, updater in self.DE_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                for p in sampler.moves[i].moves[j].gamma0.keys():
                    sampler.moves[i].moves[j].gamma0[p] *= output["DE"][i,j]
                updater.scale = output["DE"][i,j]
            else:
                for p in sampler.moves[inds].gamma0.keys():
                    sampler.moves[inds].gamma0[p] *= output["DE"][inds]
                updater.scale = output["DE"][inds]
        for inds, updater in self.Snooker_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                for p in sampler.moves[i].moves[j].gamma0.keys():
                    sampler.moves[i].moves[j].gamma0[p] *= output["Snooker"][i,j]
                updater.scale = output["Snooker"][i,j]
                """
                for key in updater.cov_function.keys():
                     sampler.moves[i].moves[j].update_scales[key] = np.diag(
                             updater.cov_function(chain[key], rowvar = False))
                """
            else:
                for p in sampler.moves[inds].gamma0.keys():
                    sampler.moves[inds].gamma0[p] *= output["Snooker"][inds]
                updater.scale = output["Snooker"][inds]
                """
                for key in updater.cov_function.keys():
                     sampler.moves[inds].update_scales[key] = np.diag(
                             updater.cov_function(chain[key], rowvar = False))
                """
        for inds, updater in self.Fisher_updaters.items():
            if isinstance(inds,tuple):    
                i,j = inds
                log_scale = output["Fisher"][i,j]
                for p in  list(sampler.moves[i].moves[j].all_proposal.keys()):
                    sampler.moves[i].moves[j].all_proposal[p].scale *= np.exp(2*log_scale)
                    sampler.moves[i].moves[j].all_proposal[p].svd *= np.exp(log_scale)
            else:
                log_scale = output["Fisher"][inds]
                for p in  list(sampler.moves[inds].all_proposal.keys()):
                    sampler.moves[inds].all_proposal[p].scale *= np.exp(2*log_scale)
                    sampler.moves[inds].all_proposal[p].svd *= np.exp(log_scale)



        
    def __call__(self, i, last, samp):
        current_it = samp.iteration
        discard = int(samp.iteration * 0.8)
        # Print the current acceptance rate
        if (current_it > self._acc_rate_step) and (current_it % self._acc_rate_step ==0):
            acc_rates_txt =[f"{np.mean(mv.acceptance_fraction[0]):.5f}" for mv in samp.moves]
            swaps_rate = "\t".join([f"{x:.5f}" for x in samp.swap_acceptance_fraction[:3]])
            with open(self.backend_dir + "/acc_rate.txt","a") as f0:
                f0.write(f"{current_it}\t" + "\t".join(acc_rates_txt) +"\t"+  swaps_rate + "\n")

        if (current_it > self._stretch_update) and (current_it % self._stretch_update == 0) and (current_it < self._stop_update):
            for inds, updater in self.Stretch_updaters.items():
                if isinstance(inds,tuple):
                    i,j = inds
                    updater(samp.moves[i].moves[j])
                else:
                    updater(samp.moves[inds])

        if (current_it > self._AM_update) and (current_it % self._AM_update == 0) and (current_it < self._stop_update):
            samples = samp.get_chain(discard = discard)
            chain = {p: samples[p][:,0,:,0].reshape(-1,samp.ndims[p]) for p in samp.branch_names}
            for inds, updater in self.AM_updaters.items():
                if isinstance(inds,tuple):
                    i,j = inds
                    updater(samp.moves[i].moves[j], chain)
                else:
                    updater(samp.moves[inds], chain)

        if (current_it > self._DE_update) and (current_it % self._DE_update == 0) and (current_it < self._stop_update):
            for inds, updater in self.DE_updaters.items():
                if isinstance(inds,tuple):
                    i,j = inds
                    updater(samp.moves[i].moves[j])
                else:
                    updater(samp.moves[inds])

        if (current_it > self._Fisher_update) and (current_it % self._Fisher_update == 0) and (current_it < self._stop_update):
            for inds, updater in self.Fisher_updaters.items():
                if isinstance(inds,tuple):
                    i,j = inds
                    updater(samp.moves[i].moves[j])
                else:
                    updater(samp.moves[inds])

        if (current_it > self._Snooker_update) and (current_it % self._Snooker_update == 0) and (current_it < self._stop_update):
            samples = samp.get_chain(discard = discard)
            chain = {p: samples[p][:,0,:,0].reshape(-1,samp.ndims[p]) for p in samp.branch_names}
            for inds, updater in self.Snooker_updaters.items():
                if isinstance(inds,tuple):
                    i,j = inds
                    updater(samp.moves[i].moves[j], chain)
                else:
                    updater(samp.moves[inds], chain)
        #TODO 
        # add the DE updater 
        # add fisher updater
        if (current_it > 1) and (current_it %50 == 0):
            # dump info 
            self._dump_move_info(samp)
        # plot the corner 
        if (current_it % self._plot_iteration ==0) and (current_it > 0):
            #breakpoint()
            samples = samp.get_chain(discard = discard)
            logl    = samp.get_log_like(discard=discard)[:,0].flatten()
            for p in samp.branch_names:
                chain = np.hstack((samples[p][:,0].reshape(-1, samp.ndims[p]), logl[:,None]))
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
