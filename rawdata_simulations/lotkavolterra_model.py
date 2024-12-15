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
from resource_model import MergedSolution

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

class BaseLotkaVolterraModel:
    """Lotka-Volterra model
    :param nS: number of species
    :param r: growth rate of species
    :param A: interaction matrix
    :param Rm: michaelis-menten constant for resource uptake"""
    def __init__(self, nS_rsd, nS_ivd, r_pool=None, A_pool=None, seed=None):
        self.nS_rsd = nS_rsd  # number of resident species
        self.nS_ivd = nS_ivd  # number of invader species
        self.nS_pool = nS_rsd + nS_ivd  # number of species in the pool
        self.nR = 1

        if seed is not None:
            np.random.seed(seed)

        if r_pool is None:
            self.__r_pool = self.generate_random_parameters('r')
        else:
            self.__r_pool = copy.deepcopy(r_pool)
        if A_pool is None:
            self.__A_pool = self.generate_random_parameters('A')
        else:
            self.__A_pool = copy.deepcopy(A_pool)

        # create a timestamp for parameters of species pool
        self.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.A_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    @property
    def r_pool(self):
        return self.__r_pool
    @property
    def A_pool(self):
        return self.__A_pool
    @r_pool.setter
    def r_pool(self, r_pool):
        self.__r_pool = r_pool
        self.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    @A_pool.setter
    def A_pool(self, A_pool):
        self.__A_pool = A_pool
        self.A_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    def generate_random_parameters(self, param):
        if param == 'r':
            # return np.random.uniform(0, 1, self.nS_pool)
            return np.concatenate([np.random.uniform(0.1, 0.5, self.nS_rsd),
                                   np.random.uniform(0.45, 0.6, self.nS_ivd)])
        elif param == 'A':
            nS = self.nS_rsd + self.nS_ivd

            catagory_raw = np.random.rand(nS, nS)
            catagory = np.zeros((nS, nS))  # 0: no interaction
            catagory[catagory_raw < 0.2] = 1  # weak interaction
            catagory[catagory_raw >= 0.6] = 2  # strong interaction
            catagory[np.eye(nS, dtype=bool)] = 3  # self-interaction

            randA = np.random.uniform(0, 1, (nS, nS))
            randA[catagory == 0] = 0  # no interaction: 0
            randA[catagory == 1] = randA[catagory == 1] * -0.6 + 0.1  # weak interaction: -0.5 - 0.1
            randA[catagory == 2] = randA[catagory == 2] * -2  # strong interaction: -2 - 0
            randA[catagory == 3] = randA[catagory == 3] * 0 - 1  # self interaction: -1

            # modify invader's impact on resident to make it stronger
            if self.nS_ivd > 0:
                randA[:self.nS_rsd, -self.nS_ivd:] = np.random.uniform(-3, 0, (self.nS_rsd, self.nS_ivd))
            return randA


class LotkaVolterraModel():
    """Lotka-Volterra model
    :param nS: number of species
    :param r: growth rate of species
    :param A: interaction matrix
    :param Rm: michaelis-menten constant for resource uptake

    dN/dt = r*N*(1 - A*N)*(R / (Rm + R))
    dR/dt = -sum(dN/dt)

    :param seed: random seed
    """
    def __init__(self, r=None, A=None, sp_idexs=None, mod_pool=None, Rm=0.1):
        # initialize either by (mod_pool, sp_idexs) or by (r, A).
        if sp_idexs is None:
            assert A is not None and r is not None, "sp_idexs is None, A and r should be provided"
            self.nS = len(r)
            assert A.shape == (self.nS, self.nS), "A should be a square matrix each dimension with the same size as r"
            self.__r = copy.deepcopy(r)
            self.__A = copy.deepcopy(A)
            self.sp_idexs = np.arange(self.nS)
            self.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            self.A_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        else:  # initialize by mod_pool and sp_idexs
            self.nS = len(sp_idexs)
            assert mod_pool is not None, "to initiate with sp_idexs, mod_pool should be BaseLotkaVolterraModel"
            assert np.max(sp_idexs) < mod_pool.nS_pool, "sp_idexs should be within the range of mod_pool.nS_pool"
            if r is None:
                self.__r = mod_pool.r_pool[sp_idexs]
                self.r_timestamp = mod_pool.r_timestamp
            else:
                assert len(r) == len(sp_idexs), "r should have the same length as sp_idexs"
                self.__r = copy.deepcopy(r)
                self.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            if A is None:
                self.__A = mod_pool.A_pool[sp_idexs, :][:, sp_idexs]
                self.A_timestamp = mod_pool.A_timestamp
            else:
                assert A.shape == (self.nS, self.nS), "A should be a square matrix each dimension with the same size as r"
                self.__A = copy.deepcopy(A)
                self.A_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            self.sp_idexs = copy.deepcopy(sp_idexs)
        
        self.Rm = Rm
        self.nR = 1  # for compatibility with ResourceModel, not used

    @property
    def r(self):
        return self.__r

    @property
    def A(self):
        return self.__A

    @r.setter
    def r(self, r):
        self.__r = r
        self.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    @A.setter
    def A(self, A):
        self.__A = A
        self.A_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    def dNdt_dRdt(self, N):
        """Return dNdt and dRdt
        """
        N = copy.deepcopy(N)
        N[N < 0] = 0  # treat negative values (occur due to numerical accuracy limit) as 0
        N = N.reshape([-1, 1])

        # dN = self.r.reshape([-1, 1]) * N * (1 + self.A @ N) #* (1 - np.sum(N) / 2)
        R = 3 - np.sum(N)
        if R <= 0:
            dN = self.r.reshape([-1, 1]) * 0
        else:
            dN = self.r.reshape([-1, 1]) * N * (1 + self.A @ N) * (R / (self.Rm + R))
        dR = np.array([0])
        return np.concatenate([dN.flatten(), dR.flatten()])

    def model(self, t, y):
        """Model for solve_ivp
        """
        N = y[:self.nS]
        # R = y[self.nS:]
        return self.dNdt_dRdt(N)

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

    def sub_community(self, sub_sp_idexes):
        """return a new community model keeping part of the species in the original model
        sp_idxes: index of species to keep
        """
        # make a copy of the original model
        mod_sub = copy.deepcopy(self)
        # sub indexing, keep parameter timestamps unchanged
        mod_sub.__r = self.__r[sub_sp_idexes]
        mod_sub.__A = self.__A[sub_sp_idexes, :][:, sub_sp_idexes]
        mod_sub.sp_idexs = mod_sub.sp_idexs[sub_sp_idexes]
        mod_sub.nS = len(sub_sp_idexes)
        return mod_sub

    def concat_community(self, other, mod_pool: BaseLotkaVolterraModel):
        """return a new community model by concatenating two models
        """
        # make a copy of the original model
        # check timestamp of A
        assert self.A_timestamp == mod_pool.A_timestamp, "concat failed: self.A timestamp different as mod_pool"
        assert other.A_timestamp == mod_pool.A_timestamp, "concat failed: other.A timestamp different as mod_pool"
        # check Rm
        assert self.Rm == other.Rm, "concat failed: Rm is not the same"
        # check if sp_idexs have shared elements
        assert np.intersect1d(self.sp_idexs, other.sp_idexs).size == 0, "concat failed: two models have shared species"
        
        # new model
        mod_cmb = LotkaVolterraModel(mod_pool=mod_pool, sp_idexs=np.concatenate([self.sp_idexs, other.sp_idexs]),
                                     r=np.concatenate([self.__r, other.__r]), Rm=self.Rm)
        # update timestamp of r. Timestamp of A is already checked and inherited from mod_pool
        if self.r_timestamp != other.r_timestamp:
            mod_cmb.r_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        else:
            mod_cmb.r_timestamp = self.r_timestamp

        return mod_cmb


if __name__ == "__main__":
    LotkaVolterraModel(sp_idexs=[0, 1], mod_pool=BaseLotkaVolterraModel(3, 2))