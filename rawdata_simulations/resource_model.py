from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
from simulation import *
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from sklearn import linear_model
from matplotlib import cm
import pickle
from scipy.interpolate import interp1d
import datetime
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

class MergedSolution:
    '''Merge multiple odesolver solutions (e.g. daily solutions in a daily dilution scenario) into one solution.'''
    def __init__(self, solutions):
        assert all(hasattr(sol, 't') and hasattr(sol, 'y') for sol in
                   solutions), "All inputs must be valid solution instances."

        # Sort solutions by the first element in their t (time) array
        sorted_solutions = sorted(solutions, key=lambda sol: sol.t[0])

        self.t_events = []
        self.status = []
        self.message = []
        self.success = []
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        for sol in sorted_solutions:
            if hasattr(sol, 't_events'): self.t_events.append(sol.t_events)
            if hasattr(sol, 'status'): self.status.append(sol.status)
            if hasattr(sol, 'message'): self.message.append(sol.message)
            if hasattr(sol, 'success'): self.success.append(sol.success)
            if hasattr(sol, 'nfev'): self.nfev += sol.nfev
            if hasattr(sol, 'njev'): self.njev += sol.njev
            if hasattr(sol, 'nlu'): self.nlu += sol.nlu

        # Merge t and y
        t, y = [], []
        for sol in sorted_solutions:
            for t_val, y_val in zip(sol.t, sol.y.T):
                t.append(t_val)
                y.append(y_val)
        self.t = np.array(t)
        self.y = np.array(y).T


class ResourceModel:
    """Resource based competition model.
    :param nS: number of species
    :param nR: number of resources
    :param fP: fraction of resources used by species (assigned from first, can overlap with fC)
    :param fC: fraction of resources that can lead to resistance (assigned from last, can overlap with fP)
    :param seed: random seed
    :param sort: whether to sort species by intrinsic growth rate

    N: population size vector, the first is for invasive species, the rest are for native species
    R: resource availability vector
    r: intrinsic growth rate vector, the first is for invasive species, the rest are for native species
    p: nS * nR matrix, preference/proportion of resource usage by species
    b: nS * nR matrix, secretion coefficient matrix,
        the species secreted resource per unit of population change
    bmax: nS vector, maximum secretion rate per species.
        If sum(b_j) > bmax_i, then b_j is scaled down to bmax_i * b_j / sum(b_j)
    c: nS * nR matrix, resistance coefficient matrix, the resistance provided by a unit of resource
    e: nS * nR matrix, efficiency coefficient matrix, the species usable energy provided by a unit of resource

    dNdt: change of population size vector dN/dt
    dRdt: change of resource availability vector dR/dt

    The model captures the following processes:
    1. Growth
    Species i grow exponentially with rate r_i if resource is available, otherwise 0
    r_i(R) is a function of resource availability R, which is used to capture resource-mediated competition
        dN_i/dt = r_i(R) * N_i * (sum_j(p_ij * R_j) > 0)
    Resource is partially supplied by the environment, and can be used or secreted by specie
    2. Resource consumption and secretion
    Resource are used according to preference/proportion matrix p, and is proportional to population change
        dR_j/dt = -sum_i(r_i * N_i * (sum_k(p_ik * R_k) > 0) * (p_ij * (R_j > 0)) / (sum_k(p_ik * (R_k > 0))))
            + sum_i(b_rel_ij * N_i * (sum_k(p_ik * R_k) > 0))
    3. Resource-mediated competition
    Some resources that species secreted are resistance substance, which can inhibit growth of some species
        r_i(R) = r_i * max(0, 1 - sum_j(c_ij * R_j))

    The Class includes the following functions:
    1. dNdt
    change of population size vector dN/dt
    2. dRdt
    change of resource availability vector dR/dt
    3. simulate(self, t, N0, R0)
    simulate the model in continuous time
    4. simulate_daily_dilution(self, t, d, N0, R0)
    simulate the model as 24h continuous growth + daily dilution
    5. generate_random_parameters(self, nS, nR)
    generate random parameters for the model, based on the number of species and resources
    """

    def __init__(self, nS, nR, fP=0.7, fC=0.6, r=None, p=None, b=None, bmax=None, c=None, e=None, seed=None, sort=True):
        self.nS = nS
        self.nR = nR

        if seed is not None:
            np.random.seed(seed)

        if r is None:
            self.r = self.generate_random_parameters('r')  # intrinsic growth rate vector
        else:
            self.r = r
        if p is None:  # preference/proportion of resource usage by species
            self.p = self.generate_random_parameters('p')
            nP = round(nR * fP)
            self.p[:, nP:] = 0  # only use the first nP resources
        else:
            self.p = p
        if b is None:  # secretion coefficient matrix
            self.b = self.generate_random_parameters('b')
            if bmax is None:  # maximum secretion rate per species
                bmax = self.generate_random_parameters('bmax')
            # normalize b so for each species it sums to bmax
            b_norm = self.b / np.sum(self.b, axis=1).reshape([-1, 1]) * bmax.reshape([-1, 1])
            self.b = b_norm
        else:
            self.b = b
        if c is None:  # resistance coefficient matrix
            self.c = self.generate_random_parameters('c')
            nC = round(nR * fC)
            self.c[:, :-nC] = 0  # only the last nC resources can lead to resistance
        else:
            self.c = c
        if e is None:  # efficiency coefficient matrix
            self.e = self.generate_random_parameters('e')
        else:
            self.e = e

        # rank self.r from high to low, and rearrange other parameters accordingly
        if sort:
            rank = np.argsort(-self.r)
            self.r = self.r[rank]
            self.p = self.p[rank]
            self.b = self.b[rank]
            self.c = self.c[rank]
            self.e = self.e[rank]

    def generate_random_parameters(self, name):
        if name == 'r':
            return np.random.uniform(0.1, 0.7, self.nS)
        elif name == 'p':
            rate = np.random.uniform(-4, 1, [self.nS, self.nR])
            rate[:, :3] = np.random.uniform(-1, 1, [self.nS, 3])
            rate[rate < 0] = 0
            return rate
        elif name == 'b':
            rate = np.random.uniform(-0.2, 0.1, [self.nS, self.nR])
            rate[rate < 0] = 0
            return rate
        elif name == 'bmax':
            return np.random.uniform(0.2, 1, self.nS)
        elif name == 'c':
            rate = np.random.uniform(-2, 1, [self.nS, self.nR])
            rate[rate < 0] = 0
            return rate
        elif name == 'e':
            # return np.random.uniform(0.7, 1, [self.nS, self.nR])
            return np.ones([self.nS, self.nR])
        else:
            raise ValueError('Invalid parameter name: {}'.format(name))

    def dNdt_dRdt(self, N, R):
        N = copy.deepcopy(N)
        N[N < 0] = 0  # treat negative values (occur due to numerical accuracy limit) as 0
        N = N.reshape([-1, 1])
        R = copy.deepcopy(R)
        R[R < 0] = 0  # treat negative values (occur due to numerical accuracy limit) as 0
        R = R.reshape([1, -1])

        # per species preference for using available resources
        p_avail = self.p * (R > 0)
        # normalize p_avail so for each species it sums to 1
        p_rel = p_avail / np.sum(p_avail, axis=1, keepdims=True)
        p_rel[np.isnan(p_rel)] = 0  # if no resource is available, no resource is used

        # growth only when resource is available, 0.01 is michaelis constant
        L = np.sum(R * (p_avail > 0), axis=1, keepdims=True) / (0.01 + np.sum(R * (p_avail > 0), axis=1, keepdims=True))
        # growth rate reduced by inhibition
        I = np.maximum(0, 1 - np.dot(R, self.c.T)).reshape([-1, 1])
        dN = self.r.reshape([-1, 1]) * N * L * I  # dN / dt

        # secretion matrix
        dR_sec = self.b * N * L * I
        # - uptake (for growth + for secretion) + secretion
        dR = (- np.sum((dN + np.sum(dR_sec, axis=1, keepdims=True)) * p_rel / self.e, axis=0)
              + np.sum(dR_sec, axis=0))
        return np.concatenate([dN.flatten(), dR.flatten()])

    def model(self, t, y):
        N = y[:self.nS]
        R = y[self.nS:]
        return self.dNdt_dRdt(N, R)

    def simulate(self, t, N0, R0, thre_N=1e-6, thre_R=0, **kwargs):
        # if kwargs doesn't specify atol and rtol, use default values
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-6
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

        y0 = np.concatenate([N0, R0])
        if len(t) == 2:
            sol = solve_ivp(self.model, [t[0], t[-1]], y0, **kwargs)
        else:
            sol = solve_ivp(self.model, [t[0], t[-1]], y0, t_eval=t, **kwargs)

        # if values are below threshold, set them to 0
        # for N the threshold is thre_N, for R the threshold is thre_R
        sol.y[:self.nS, :][sol.y[:self.nS, :] < thre_N] = 0
        sol.y[self.nS:, :][sol.y[self.nS:, :] < thre_R] = 0
        return sol

    def simulate_daily_dilution(self, nday, d, N0, R0, T=24, t=None, thre_N=1e-6, thre_R=0, **kwargs):
        # if kwargs doesn't specify atol and rtol, use default values
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-6
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

        sol_list = np.empty((nday,), dtype=object)  # Preallocate array for solutions
        if t is None:
            t_evals = np.arange(nday).reshape([-1, 1]) * T + np.array([0, T]).reshape([1, -1])
        else:
            t_evals = np.arange(nday).reshape([-1, 1]) * T + np.array(t).reshape([1, -1])

        y0 = np.concatenate([N0, R0])
        for day in range(nday):
            y0[y0 < 0] = 0  # set negative values (occur due to numerical accuracy limit) to 0
            # one_day_sol = solve_ivp(self.model, [day * T, (day + 1) * T], y0, t_eval=t_evals[day], **kwargs)
            one_day_sol = self.simulate(t_evals[day], y0[:self.nS], y0[self.nS:],
                                        thre_N=thre_N, thre_R=thre_R, **kwargs)
            sol_list[day] = one_day_sol
            y0 = one_day_sol.y[:, -1] * 1
            y0[:self.nS] *= d
            y0[self.nS:] = d * y0[self.nS:] + (1 - d) * R0
        return MergedSolution(sol_list)

    def concat_community(self, other):
        """add new species into the model, return a new model as an object of class ResourceModel
        the new species are described as an object of class ResourceModel
        new species are appended to the end of the original model
        """
        nS = self.nS + other.nS
        # check if the number of resources are the same
        if self.nR != other.nR:
            raise ValueError("The two models must have the same number of resources")
        nR = self.nR
        mod_new = ResourceModel(nS, nR, r=np.concatenate([self.r, other.__r]),
                                p=np.concatenate([self.p, other.p], axis=0),
                                b=np.concatenate([self.b, other.b], axis=0),
                                c=np.concatenate([self.c, other.c], axis=0),
                                e=np.concatenate([self.e, other.e], axis=0), sort=False)
        return mod_new

    def sub_community(self, sp_idxes):
        """return a new community model keeping part of the species in the original model
        sp_idxes: index of species to keep
        """
        # make a copy of the original model
        mod_sub = ResourceModel(len(sp_idxes), self.nR, r=self.r[sp_idxes], p=self.p[sp_idxes],
                                b=self.b[sp_idxes], c=self.c[sp_idxes], e=self.e[sp_idxes], sort=False)
        return mod_sub

def sim_wavespeed(func, m_list, nday=40, nwell=50, cutoffs=[0.1, 0.9], insta=False):
    """
    if insta==True, also return the max and min of instantaneous speeds after day 10.
    given invader response function (invader % after interaction vs. invader % before interaction), simulate invasion experiment and return wavespeed
    """
    wave_speeds = np.zeros([len(cutoffs), len(m_list)])
    if insta == True:
        if nday < 20:
            print('to calculate instanteneous wave speeds, nday must > 20')
            insta = False
        else:
            wave_min_speeds = np.zeros(len(m_list))
            wave_max_speeds = np.zeros(len(m_list))
    for im, m in enumerate(m_list):
        usedays = min(nday - 5, nwell)

        series = np.array(mig_rowseries(g=func, m=m, day=nday, well=[nwell, nwell], draw=False))  # nday * (6+nwell)
        for ii, cutoff in enumerate(cutoffs):
            wave_fronts = np.array([np.where(s > cutoff)[0].max() for s in series])  # invation front, with >0.1 invader
            popt, pcov = curve_fit(lambda x, k: k * x, range(nday - usedays, nday), wave_fronts[-usedays:] - nwell + 1,
                                   p0=[0], method='lm')
            wave_speeds[ii, im] = popt
        if insta == True:
            insta_speed = np.sum(series[11:, :] - series[10:-1, :], axis=1)
            wave_max_speeds[im] = insta_speed.max()
            wave_min_speeds[im] = insta_speed.min()
            wave_speeds[0, im] = insta_speed.mean()

    if insta == True:
        return wave_speeds, wave_max_speeds, wave_min_speeds
    else:
        return wave_speeds


class InvPairModel(ResourceModel):
    """inherit from ResourceModel, but specify the invader and native species,
        with their equilibrium population sizes when alone"""

    def __init__(self, mod, i_inv, N_equi, R0=None, t=24):
        """
        :param mod: species interaction model of class ResourceModel
        :param i_inv: index of invader
        :param N_equi: equilibrium population size vector of invader alone
            + equilibrium population size vector of community without invader
        :param R0: initial resource availability vector, supplied every dispersal&dilution cycle
        :param t: time of a dispersal&dilution cycle
        """
        # if mod is a ResourceModel, copy its attributes and values
        if isinstance(mod, ResourceModel):
            for key, value in mod.__dict__.items():
                setattr(self, key, value)
        else:
            raise ValueError('mod must be a ResourceModel')

        self.i_inv = i_inv  # index of invader
        # equilibrium population size of invader, vector of length nspecies, with 0 for non-invader species
        N_inv = np.zeros(self.nS)
        N_inv[i_inv] = N_equi[i_inv]
        self.N_inv = N_inv
        # equilibrium composition of community without invader, vector of length nspecies, with 0 for invader species
        N_cm = N_equi.copy()
        N_cm[i_inv] = 0
        self.N_cm = N_cm

        if R0 is None:
            R0 = np.zeros(self.nR)
            R0[0] = 1
        self.R0 = R0
        self.t = t

if __name__ == "__main__":
    nrsd = 100
    nivd = 100
    for irsd in range(20, nrsd):
        try:
            with open(f'mod0rsd_20230812_{irsd}.pkl', 'rb') as f:
                data = pickle.load(f)
                mod_rsd = data['mod_rsd']
                sol_rsd = data['sol_rsd']
        except FileNotFoundError:
            print(f'mod0rsd_20230812_{irsd}.pkl not found, skiped')
            continue
        print(f'start irsd={irsd}')
        comp_diff_ivds(mod_rsd, sol_rsd, irsd, nivd)
