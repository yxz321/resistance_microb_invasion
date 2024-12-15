import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from resource_model import ResourceModel
from lotkavolterra_model import LotkaVolterraModel, BaseLotkaVolterraModel
from simulation import mig_rowseries, mig_rowseries_v2, mig_rowseries_2dim
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import pickle
import warnings

def modgen_resident(seed, nS=50, nR=30, df=200, c_scale=20, save2file=None):
    """Generate resident model
    :param seed: random seed
    :param nS: number of species
    :param nR: number of resources
    :param df: dilution factor
    :param c_scale: scaling factor for c (inhibitory coefficient of resources)
    :param save2file: file name to save the model and solution

    :return: dictionary containing the model and solution
    """
    # mod_base: base community with nS possible species
    np.random.seed(seed)
    mod_base = ResourceModel(nS, nR, fP=np.random.uniform(0.2, 0.8), fC=np.random.uniform(0.2, 0.8), seed=seed)
    mod_base.c *= c_scale

    # sol_base
    R0 = np.zeros(mod_base.nR)
    R0[:3] = 1/3  # main carbon source
    N0 = np.ones(mod_base.nS) * 0.001
    sol_base = mod_base.simulate_daily_dilution(30, 1 / df, N0, R0)
    sol_base_daily = sol_base.y[:, sol_base.t % 24 == 0]

    # mod_rsd: resident community with only species with equilibrium population > 1e-6
    rmn_sp_idxes = np.where(sol_base.y[:nS, -1] > 1e-6)[0]  # species with equilibrium population > 1e-6
    mod_rsd = mod_base.sub_community(rmn_sp_idxes)

    # sol_rsd
    N0 = sol_base.y[rmn_sp_idxes, -1] / df
    R0 = R0 * (1 - 1 / df) + sol_base.y[nS:, -1] / df
    sol_rsd = mod_rsd.simulate_daily_dilution(10, 1 / df, N0, R0)
    sol_rsd_daily = sol_rsd.y[:, sol_rsd.t % 24 == 0]

    var_list = ['mod_base', 'sol_base', 'sol_base_daily', 'mod_rsd', 'sol_rsd', 'sol_rsd_daily', 'N0', 'R0']
    data = {}
    for var in var_list:
        data[var] = locals()[var]
    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)
    return data


def modgen_invader(seed, nR=30, df=200, c_scale=20, save2file=None):
    """ Generate invader model
    :param seed: random seed
    :param nR: number of resources
    :param df: dilution factor
    :param c_scale: scaling factor for c (inhibitory coefficient of resources)
    :param save2file: file name to save the model and solution

    :return: dictionary containing the model and solution
    """
    # mod_ivd
    seed += 7182818
    np.random.seed(seed)
    mod_ivd = ResourceModel(1, nR, fP=np.random.uniform(0.2, 1), fC=np.random.uniform(0, 0.8), seed=seed)
    # rescale r to be in [0.6, 0.8] instead of [0.1, 0.7] as default
    tmp = (mod_ivd.r - 0.1) / (0.7 - 0.1)
    mod_ivd.r = 0.6 + tmp * (0.8 - 0.6)
    mod_ivd.p[0, :3] += 0.1  # increase the uptake preference of the main carbon sources
    mod_ivd.c *= c_scale

    # sol_ivd
    R0 = np.zeros(mod_ivd.nR)
    R0[:3] = 1/3  # main carbon source
    N0 = [0.001]
    sol_ivd = mod_ivd.simulate_daily_dilution(10, 1 / df, N0, R0)
    sol_ivd_daily = sol_ivd.y[:, sol_ivd.t % 24 == 0]

    var_list = ['mod_ivd', 'sol_ivd', 'sol_ivd_daily', 'N0', 'R0']
    data = {}
    for var in var_list:
        data[var] = locals()[var]
    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)
    return data


def lv_modgen_resident(mod_pool, seed, nS=30, rsd_range=None, df=200, save2file=None):
    """Generate resident model
    :param mod_pool: LotkaVolterraModel object
    :param seed: random seed
    :param nS: number of species
    :param df: dilution factor

    :return: dictionary containing the model and solution
    """
    # mod_base: base community with nS possible species
    np.random.seed(seed)
    if rsd_range is None:
        rsd_range = np.arange(mod_pool.nS_rsd)
    base_sp_idxes = np.random.permutation(rsd_range)[:nS]
    mod_base = LotkaVolterraModel(sp_idexs=base_sp_idxes, mod_pool=mod_pool)

    # sol_base
    R0 = np.array([1])
    N0 = np.ones(mod_base.nS) * 0.001
    sol_base = mod_base.simulate_daily_dilution(50, 1 / df, N0, R0)
    sol_base_daily = sol_base.y[:, sol_base.t % 24 == 0]

    sp_above_thre = np.where(sol_base.y[:nS, -1] > 1e-6)[0]  # species with equilibrium population > 1e-6
    mod_rsd = mod_base.sub_community(sp_above_thre)

    # sol_rsd
    N0 = sol_base.y[sp_above_thre, -1] / df
    R0 = R0 * (1 - 1 / df) + sol_base.y[nS:, -1] / df
    sol_rsd = mod_rsd.simulate_daily_dilution(10, 1 / df, N0, R0)
    sol_rsd_daily = sol_rsd.y[:, sol_rsd.t % 24 == 0]

    var_list = ['mod_base', 'sol_base', 'sol_base_daily', 'mod_rsd', 'sol_rsd', 'sol_rsd_daily', 'N0', 'R0']
    data = {}
    for var in var_list:
        data[var] = locals()[var]
    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)
    return data


def lv_modgen_invader(mod_pool, seed, ivd_range=None, df=200, save2file=None):
    """ Generate invader model
    :param mod_pool: LotkaVolterraModel object
    :param seed: seed for picking up species, in backwards order so that 0 is the last species in the pool
    :param df: dilution factor
    :param save2file: file name to save the model and solution

    :return: dictionary containing the model and solution
    """
    # mod_ivd
    np.random.seed(seed)
    if ivd_range is None:
        ivd_range = np.arange(mod_pool.nS_rsd, mod_pool.nS_pool)  # equal to: [-nS_ivd:]
    ivd_sp_idxes = np.random.permutation(ivd_range)[:1]
    mod_ivd = LotkaVolterraModel(sp_idexs=ivd_sp_idxes, mod_pool=mod_pool)

    # sol_ivd
    R0 = np.array([1])
    N0 = [0.001]
    sol_ivd = mod_ivd.simulate_daily_dilution(10, 1 / df, N0, R0)
    sol_ivd_daily = sol_ivd.y[:, sol_ivd.t % 24 == 0]

    var_list = ['mod_ivd', 'sol_ivd', 'sol_ivd_daily', 'N0', 'R0']
    data = {}
    for var in var_list:
        data[var] = locals()[var]
    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)
    return data


def comp_curve(mod_ivd, NR_ivd, mod_rsd, NR_rsd, mod_cmb=None, R_sup=None, mix_frac=None, df=200, T=24, randseed=None, save2file=False):
    """ Competition curve with randomized species abundance and save original abundance data
    """
    if mod_cmb is None:
        mod_cmb = mod_ivd.concat_community(mod_rsd)
    N_ivd = np.zeros(mod_cmb.nS)
    N_ivd[:mod_ivd.nS] = NR_ivd[:mod_ivd.nS]
    N_rsd = np.zeros(mod_cmb.nS)
    N_rsd[mod_ivd.nS:] = NR_rsd[:mod_rsd.nS]
    R_ivd, R_rsd = NR_ivd[mod_ivd.nS:], NR_rsd[mod_rsd.nS:]


    if R_sup is None:
        R_sup = np.zeros(mod_cmb.nR)
        n_mainC = np.min((3, mod_cmb.nR))
        R_sup[:n_mainC] = 1 / n_mainC  # main carbon source, externally supplied
    if mix_frac is None:
        mix_frac = np.array([0, 0.01, 0.1, 0.4, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]) / 100

    sol_arr = np.empty(len(mix_frac)).astype('object')
    sol_arr.fill(np.nan)
    fin_abd_allNR = np.zeros([mod_cmb.nS + mod_cmb.nR, len(mix_frac)])
    fin_abd_allNR.fill(np.nan)
    ini_abd_allNR = np.zeros([mod_cmb.nS + mod_cmb.nR, len(mix_frac)])
    ini_abd_allNR.fill(np.nan)
    for ii in range(len(mix_frac)):
        f = mix_frac[ii]
        if randseed:
            np.random.seed(randseed)
            N_rsd_rand = np.random.rand(mod_cmb.nS)
            N_rsd_rand[-1] = 0
            N_rsd_rand = N_rsd_rand / N_rsd_rand.sum() * N_rsd.sum()
            N_rsd = copy.deepcopy(N_rsd_rand)
        N0 = (N_ivd * f + N_rsd * (1 - f)) / df
        R0 = (R_ivd * f + R_rsd * (1 - f)) / df + R_sup * (1 - 1 / df)
        ini_abd_allNR[:mod_cmb.nS, ii] = N0
        ini_abd_allNR[mod_cmb.nS:, ii] = R0
        sol = mod_cmb.simulate_daily_dilution(1, 1 / df, N0, R0, T=T, rtol=1e-3,
                                              atol=1e-6)  # initial transfer with different equilibrium community fraction
        fin_abd_allNR[:, ii] = sol.y[:, -1]

    var_list = ['mix_frac', 'ini_abd_allNR', 'fin_abd_allNR', 'mod_cmb', 'R_sup', 'df', 'randseed']
    data = {}
    for var in var_list:
        data[var] = locals()[var]
    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)

    return data

# # old version
# def sim_invasion(mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list, mod_cmb=None, R_sup=None, df=200, T=24, nday=20, nwell=[10, 10],
#                  m_scalers=[1/2, 1/2], parallel_nworker=0, save2file=False):
#     """
#     simulate invasion experiment by ode
#     assymatic dispersal rate
#     :param imod: species interaction and invasion model, of class InvPairModel
#         imod.N_inv: equilibrium population size of invader, vector of length nspecies, with 0 for non-invader species
#         imod.N_cm: equilibrium composition of community without invader, vector of length nS, with 0 for invader species
#         imod.i_inv: index of invader
#     :param m_list: list of dispersal rates simulated. Use in combination with m_scalers, default 1/2*m for symmetric dispersal.
#     :param dil: dilution rate, e.g. 1/200
#     :param nday: number of days simulated
#     :param nwell: number of wells simulated
#     :param m_scalers: scaling factors [left, right] for m, e.g. [1/2, 1/2] for symmetric dispersal, [0, 1] for one-way dispersal from left to right
#     :return: np.array of size len(m_list) * nday * nwell * nspecies
#
#     given species interaction model, simulate invasion experiment for each dispersal rate m in m_list
#         and return the population size of each species in each well at each day
#     1. Initial distribution
#     nwell[0] wells on the left side with only invader, population size per species is mod.N_inv
#     nwell[1] wells on the right side with only native species, population size per species is mod.N_cm
#     2. Dispersal and dilution
#     Each day, each well disperses m/2 to each neighbor, and 1-m to itself
#     The leftmost well disperse m/2 to right, and 1-m/2 to itself
#     The rightmost well disperse m/2 to left, and 1-m/2 to itself
#     After dispersal, each well is diluted by multiplying dilution rate dil
#     The result is a nS * nwell[0]+nwell[1] matrix, with each column being the population size of each species in each well
#     3. Interaction
#     After dispersal and dilution, species in each well interact with each other according to sol = imod.simulate(...)
#         N0 comes from the result of dispersal and dilution
#         R0 is imod.R0
#         t is imod.t
#     For each well, sol.y[:, -1] is the population size of each species after interaction, and is used as N0 of the current well for the next day
#     sol.y[:, -1] for each well for each day is recorded in the result matrix
#     4. Repeat 2 and 3 for nday days
#     """
#
#     # initial distribution
#     m_scalers = np.array(m_scalers)
#     if mod_cmb is None:
#         mod_cmb = mod_ivd.concat_community(mod_rsd)
#     N_ivd = np.zeros(mod_cmb.nS)
#     N_ivd[:mod_ivd.nS] = NR_ivd[:mod_ivd.nS]
#     N_rsd = np.zeros(mod_cmb.nS)
#     N_rsd[mod_ivd.nS:] = NR_rsd[:mod_rsd.nS]
#     R_ivd, R_rsd = NR_ivd[mod_ivd.nS:], NR_rsd[mod_rsd.nS:]
#     if R_sup is None:
#         R_sup = np.zeros(mod_cmb.nR)
#         n_mainC = np.min((3, mod_cmb.nR))
#         R_sup[:n_mainC] = 1 / n_mainC  # main carbon source, externally supplied
#
#
#     N0 = np.zeros([mod_cmb.nS, sum(nwell)])
#     N0[:, :nwell[0]] = np.tile(N_ivd.reshape(-1, 1), nwell[0])
#     N0[:, nwell[0]:] = np.tile(N_rsd.reshape(-1, 1), nwell[1])
#     R0 = np.zeros([mod_cmb.nR, sum(nwell)])
#     R0[:, :nwell[0]] = np.tile(R_ivd.reshape(-1, 1), nwell[0])
#     R0[:, nwell[0]:] = np.tile(R_rsd.reshape(-1, 1), nwell[1])
#     N0_ini = copy.deepcopy(N0)
#     R0_ini = copy.deepcopy(R0)
#
#     # define data structure to store results
#     result = np.zeros([len(m_list), 1 + nday, sum(nwell), mod_cmb.nS + mod_cmb.nR])
#     result[:, 0, :, :mod_cmb.nS] = np.tile(N0.T[np.newaxis, :, :], [len(m_list), 1, 1])
#     result[:, 0, :, mod_cmb.nS:] = np.tile(R0.T[np.newaxis, :, :], [len(m_list), 1, 1])
#
#     # iterate over migration rates
#     for im, m in enumerate(m_list):
#         N0, R0 = copy.deepcopy(N0_ini), copy.deepcopy(R0_ini)
#         m_l, m_r = m * m_scalers
#         # iterate over days
#         for iday in range(nday):
#             # dispersal and dilution
#             N_mixed = np.zeros([mod_cmb.nS, sum(nwell)])
#             N_mixed[:, 0] = N0[:, 0] * (1 - m_l) + N0[:, 1] * m_l  # leftmost well
#             N_mixed[:, -1] = N0[:, -1] * (1 - m_r) + N0[:, -2] * m_r  # rightmost well
#             N_mixed[:, 1:-1] = N0[:, 1:-1] * (1 - m_l - m_r) + N0[:, :-2] * m_r + N0[:, 2:] * m_l  # other wells
#             N_mixed = N_mixed * 1 / df
#             R_mixed = np.zeros([mod_cmb.nR, sum(nwell)])
#             R_mixed[:, 0] = R0[:, 0] * (1 - m_l) + R0[:, 1] * m_l
#             R_mixed[:, -1] = R0[:, -1] * (1 - m_r) + R0[:, -2] * m_r
#             R_mixed[:, 1:-1] = R0[:, 1:-1] * (1 - m_l - m_r) + R0[:, :-2] * m_r + R0[:, 2:] * m_l
#             R_mixed = R_mixed * 1 / df + R_sup.reshape([-1, 1]) * (1 - 1 / df)
#
#             # interaction
#             N0 = np.zeros([mod_cmb.nS, sum(nwell)])  # to store N0 for next day
#             R0 = np.zeros([mod_cmb.nR, sum(nwell)])  # to store R0 for next day
#             # iterate over wells
#             if parallel_nworker <= 1:
#                 for iw in range(sum(nwell)):
#                     # simulate interaction
#                     sol = mod_cmb.simulate([0, T], N_mixed[:, iw], R_mixed[:, iw])
#                     # record result
#                     result[im, iday, iw, :] = sol.y[:, -1]
#                     # use result as N0 for next day
#                     N0[:, iw] = sol.y[:mod_cmb.nS, -1]
#                     R0[:, iw] = sol.y[mod_cmb.nS:, -1]
#             else:
#                 futures = np.empty(sum(nwell)).astype('object')
#                 with ProcessPoolExecutor(max_workers=parallel_nworker) as executor:
#                     for iw in range(sum(nwell)):
#                         futures[iw] = executor.submit(mod_cmb.simulate, [0, T], N_mixed[:, iw], R_mixed[:, iw])
#                     for iw in range(sum(nwell)):
#                         sol = futures[iw].result()
#                         result[im, iday, iw, :] = sol.y[:, -1]
#                         N0[:, iw] = sol.y[:mod_cmb.nS, -1]
#                         R0[:, iw] = sol.y[mod_cmb.nS:, -1]
#     if save2file:
#         var_list = ['result', 'm_list', 'mod_ivd', 'mod_rsd', 'R_sup', 'df', 'T', 'nday', 'nwell']
#         data = {}
#         for var in var_list:
#             data[var] = locals()[var]
#         with open(save2file, 'wb') as f:
#             pickle.dump(data, f)
#
#     return result


def sim_invasion(mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list, mod_cmb=None, R_sup=None, df=200, T=24, nday=20,
                       nwell=[10, 10], simple_out=True,
                       m_scalers=[1 / 2, 1 / 2], parallel_nworker=0, rand_m=0, rand_K=0, rand_r=0, save2file=False):
    """
    simulate invasion experiment by ode
    assymatic dispersal rate
    :param imod: species interaction and invasion model, of class InvPairModel
        imod.N_inv: equilibrium population size of invader, vector of length nspecies, with 0 for non-invader species
        imod.N_cm: equilibrium composition of community without invader, vector of length nS, with 0 for invader species
        imod.i_inv: index of invader
    :param m_list: list of dispersal rates simulated. Use in combination with m_scalers, default 1/2*m for symmetric dispersal.
    :param dil: dilution rate, e.g. 1/200
    :param nday: number of days simulated
    :param nwell: number of wells simulated
    :param simple_out: if True, only return the final population size (nmig * nwell * nspecies), otherwise return a dictionary containing key variables
    :param m_scalers: scaling factors [left, right] for m, e.g. [1/2, 1/2] for symmetric dispersal, [0, 1] for one-way dispersal from left to right
    :param rand_m: relative amplitude of random noise for m
    :param rand_K: relative amplitude of random noise for K
    :param rand_r: relative amplitude of random noise for r
    :param parallel_nworker: number of parallel workers for simulation
    :param save2file: file name to save the model and solution
    :return: np.array of size len(m_list) * nday * nwell * nspecies

    given species interaction model, simulate invasion experiment for each dispersal rate m in m_list
        and return the population size of each species in each well at each day
    1. Initial distribution
    nwell[0] wells on the left side with only invader, population size per species is mod.N_inv
    nwell[1] wells on the right side with only native species, population size per species is mod.N_cm
    2. Dispersal and dilution
    Each day, each well disperses m/2 to each neighbor, and 1-m to itself
    The leftmost well disperse m/2 to right, and 1-m/2 to itself
    The rightmost well disperse m/2 to left, and 1-m/2 to itself
    After dispersal, each well is diluted by multiplying dilution rate dil
    The result is a nS * nwell[0]+nwell[1] matrix, with each column being the population size of each species in each well
    3. Interaction
    After dispersal and dilution, species in each well interact with each other according to sol = imod.simulate(...)
        N0 comes from the result of dispersal and dilution
        R0 is imod.R0
        t is imod.t
    For each well, sol.y[:, -1] is the population size of each species after interaction, and is used as N0 of the current well for the next day
    sol.y[:, -1] for each well for each day is recorded in the result matrix
    4. Repeat 2 and 3 for nday days
    """

    # initial distribution
    m_scalers = np.array(m_scalers)
    if mod_cmb is None:
        mod_cmb = mod_ivd.concat_community(mod_rsd)
    N_ivd = np.zeros(mod_cmb.nS)
    N_ivd[:mod_ivd.nS] = NR_ivd[:mod_ivd.nS]
    N_rsd = np.zeros(mod_cmb.nS)
    N_rsd[mod_ivd.nS:] = NR_rsd[:mod_rsd.nS]
    R_ivd, R_rsd = NR_ivd[mod_ivd.nS:], NR_rsd[mod_rsd.nS:]
    if R_sup is None:
        R_sup = np.zeros(mod_cmb.nR)
        n_mainC = np.min((3, mod_cmb.nR))
        R_sup[:n_mainC] = 1 / n_mainC  # main carbon source, externally supplied

    N0 = np.zeros([mod_cmb.nS, sum(nwell)])
    N0[:, :nwell[0]] = np.tile(N_ivd.reshape(-1, 1), nwell[0])
    N0[:, nwell[0]:] = np.tile(N_rsd.reshape(-1, 1), nwell[1])
    R0 = np.zeros([mod_cmb.nR, sum(nwell)])
    R0[:, :nwell[0]] = np.tile(R_ivd.reshape(-1, 1), nwell[0])
    R0[:, nwell[0]:] = np.tile(R_rsd.reshape(-1, 1), nwell[1])
    N0_ini = copy.deepcopy(N0)
    R0_ini = copy.deepcopy(R0)

    # define data structure to store results
    result = np.zeros([len(m_list), 1 + nday, sum(nwell), mod_cmb.nS + mod_cmb.nR])
    result[:, 0, :, :mod_cmb.nS] = np.tile(N0.T[np.newaxis, :, :], [len(m_list), 1, 1])
    result[:, 0, :, mod_cmb.nS:] = np.tile(R0.T[np.newaxis, :, :], [len(m_list), 1, 1])

    # generate random noise
    m_rand = 1 + 2 * rand_m * (
                np.random.rand(1, sum(nwell)) - rand_m)  # perturb m with random noise of relative amplitude rand_m
    K_rand = 1 + 2 * rand_K * (
                np.random.rand(1, sum(nwell)) - rand_K)  # perturb K with random noise of relative amplitude rand_K
    r_rand = 1 + 2 * rand_r * (np.random.rand(mod_cmb.nS,
                                              sum(nwell)) - rand_r)  # perturb r with random noise of relative amplitude rand_r

    # iterate over migration rates
    for im, m in enumerate(m_list):
        N0, R0 = copy.deepcopy(N0_ini), copy.deepcopy(R0_ini)
        NR0 = np.concatenate([N0, R0], axis=0)
        m_l, m_r = np.array(m_scalers).reshape([-1, 1]) * m * m_rand
        # iterate over days
        for iday in range(nday):
            # dispersal
            NR_mixed = np.zeros([mod_cmb.nS + mod_cmb.nR, sum(nwell)])
            NR_mixed[:, 0] = NR0[:, 0] * (1 - m_l[1]) + NR0[:, 1] * m_l[1]  # leftmost well
            NR_mixed[:, -1] = NR0[:, -1] * (1 - m_r[-2]) + NR0[:, -2] * m_r[-2]  # rightmost well
            NR_mixed[:, 1:-1] = NR0[:, 1:-1] * (1 - m_l[1:-1] - m_r[1:-1]) + NR0[:, :-2] * m_r[:-2] + NR0[:, 2:] * m_l[
                                                                                                                   2:]  # other wells
            # dilution
            NR_sup = np.zeros([mod_cmb.nS + mod_cmb.nR, 1])
            NR_sup[mod_cmb.nS:, 0] = R_sup
            NR_mixed = NR_mixed * 1 / df + NR_sup * (1 - 1 / df)

            # interaction
            NR0 = np.zeros([mod_cmb.nS + mod_cmb.nR, sum(nwell)])  # to store N0, R0 for next day
            # iterate over wells
            if parallel_nworker <= 1:
                for iw in range(sum(nwell)):
                    mod_cmb_curr = copy.deepcopy(mod_cmb)
                    mod_cmb_curr.A = mod_cmb.A / K_rand[0, iw]
                    mod_cmb_curr.r = mod_cmb.r * r_rand[:, iw]
                    # simulate interaction
                    sol = mod_cmb_curr.simulate([0, T], NR_mixed[:mod_cmb.nS, iw], NR_mixed[mod_cmb.nS:, iw])
                    # record result
                    result[im, iday, iw, :] = sol.y[:, -1]
                    # use result as N0 for next day
                    NR0[:, iw] = sol.y[:, -1]
            else:
                futures = np.empty(sum(nwell)).astype('object')
                with ProcessPoolExecutor(max_workers=parallel_nworker) as executor:
                    for iw in range(sum(nwell)):
                        mod_cmb_curr = copy.deepcopy(mod_cmb)
                        mod_cmb_curr.A = mod_cmb.A / K_rand[0, iw]
                        mod_cmb_curr.r = mod_cmb.r * r_rand[:, iw]
                        futures[iw] = executor.submit(mod_cmb_curr.simulate, [0, T], NR_mixed[:mod_cmb.nS, iw],
                                                      NR_mixed[mod_cmb.nS:, iw])
                    for iw in range(sum(nwell)):
                        sol = futures[iw].result()
                        result[im, iday, iw, :] = sol.y[:, -1]
                        NR0[:, iw] = sol.y[:, -1]

    var_list = ['result', 'm_list', 'mod_ivd', 'mod_rsd', 'R_sup', 'df', 'T', 'nday', 'nwell', 'm_rand', 'K_rand',
                'r_rand']
    data = {}
    for var in var_list:
        data[var] = locals()[var]

    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump(data, f)
    if simple_out:
        return result
    else:
        return data


def predinv_diff_m(ini_frac, fin_frac, ini_mix_frac=None, m_list=None, Nivd2Nrsd=None,
                   m_scalers=[1/2, 1/2], nday=20, nwell=[10, 10],
                   interp_method='linear', save2file=None):
    """predict invasion from interaction curve for different migration rate
    :param ini_frac: list-like, initial fraction of invader and resident community
    :param fin_frac: list-like, final fraction of invader and resident community

    To account for different population size of invader and resident community, either ini_mix_frac or Nivd2Nrsd should be given
    :param ini_mix_frac: list-like, initial mixing fraction ("liquid" volume) of invader and resident community,
        can be calculated from ini_frac and Nivd2Nrsd
    :param Nivd2Nrsd: ratio of invader population size vs. resident community size

    :param m_list: list-like, migration rate list
    :param m_scalers: list-like, scaling factors [left, right] for m, e.g. [1/2, 1/2] for symmetric dispersal, [0, 1] for one-way dispersal from left to right
    :param nday: number of days simulated
    :param nwell: list of length 2: [n_ivd, n_rsd]. Initiate with n_ivd wells of invader followed by n_rsd wells of resident community
    :param interp_method: method used to interpolate the interaction curve, either 'linear' or 'pchip'
    :param save2file: file name to save the model and solution, if None, do not save
    """
    if m_list is None:
        m_list = np.array([0.002, 0.008, 0.02, 0.038, 0.06, 0.1, 0.2, 0.4])

    if ini_mix_frac is None and Nivd2Nrsd is None:
        raise (ValueError('either ini_mix_frac or Nivd2Nrsd should be given'))
    # ratio of invader population size vs. resident community size
    if Nivd2Nrsd is None:
        Nivd2Nrsd = (ini_frac[1] * (1 - ini_mix_frac[1])) / ((1 - ini_frac[1]) * ini_mix_frac[1])
    f2mixf = lambda x: x / (x + Nivd2Nrsd * (1 - x))
    mixf2f = lambda x: Nivd2Nrsd * x / (Nivd2Nrsd * x + (1 - x))
    fin_mix_frac = f2mixf(fin_frac)
    if ini_mix_frac is None:
        ini_mix_frac = f2mixf(ini_frac)

    # interaction curve used to predict invasion
    if interp_method == 'linear':
        fun = interp1d(ini_mix_frac, fin_mix_frac)
    elif interp_method == 'pchip':
        fun = PchipInterpolator(ini_mix_frac, fin_mix_frac)
    else:
        raise ValueError('interp_method should be either linear or pchip')
    result = np.empty([len(m_list), nday, sum(nwell)])

    # simulate invasion for different migration rates
    nmig = len(m_list)
    for im in range(nmig):
        m = m_list[im]
        series = np.array(mig_rowseries_v2(g=fun, m=m, m_scalers=m_scalers, day=nday, well=nwell, draw=False))  # nday * sum(nwell)
        result[im, :, :] = mixf2f(series)  # project back to fraction of abundance of invader

    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump({'result': result, 'm_list': m_list, 'ini_mix_frac': ini_mix_frac,
                         'ini_abd_frac': ini_frac, 'fin_abd_frac': fin_frac}, f)

    return result


def predinv_diff_m_2dim(ini_abd, fin_abd, m_list, nday=20, nwell=[10, 10], save2file=False):
    """predict invasion from 2-dimensional interaction curve (with both invader and resident abundance) for different migration rate
    ini_abd: 2 * nmixfrac, initial abundance of invader and resident community
        The first row is invader, the second column is resident community
        If more than 2 rows, the 2nd to last rows are summed to get the initial abundance of resident community
    fin_abd: 2 * nmixfrac, final abundance of invader and resident community
        The first row is invader, the second column is resident community
        If more than 2 rows, the 2nd to last rows are summed to get the final abundance of resident community
    m_list: migration rate list
    nday: number of days simulated
    nwell: number of wells simulated, for invader and resident community
    short_return: if True, return a short version of result

    short return: return a 3-dimension array of size nmig * nday * sum(nwell) * 2
    long return: tbd"""

    # process input data
    if ini_abd.shape[0] != fin_abd.shape[0]:
        warnings.warn('ini_abd and fin_abd have different number of species')
    if ini_abd.shape[1] != fin_abd.shape[1]:
        raise ValueError('ini_abd and fin_abd have different number of mixing fractions')
    if ini_abd.shape[1] > 2:
        ini_abd_allN = copy.deepcopy(ini_abd)
        ini_abd = np.zeros([2, ini_abd.shape[1]])
        ini_abd[0, :] = ini_abd_allN[0, :]
        ini_abd[1, :] = ini_abd_allN[1:, :].sum(axis=0)
        warnings.warn('ini_abd has more than 2 rows, summed 1: to get resident abundance')
    if fin_abd.shape[1] > 2:
        fin_abd_allN = copy.deepcopy(fin_abd)
        fin_abd = np.zeros([2, fin_abd.shape[1]])
        fin_abd[0, :] = fin_abd_allN[0, :]
        fin_abd[1, :] = fin_abd_allN[1:, :].sum(axis=0)
        warnings.warn('fin_abd has more than 2 rows, summed 1: to get resident abundance')

    # functions from initial invader fraction to final invader and resident abundance after 1 day interaction
    ini_frac = ini_abd[0, :] / ini_abd.sum(axis=0)
    f2ivd = interp1d(ini_frac, fin_abd[0, :])
    f2rsd = interp1d(ini_frac, fin_abd[1, :])

    # predict invasion for different migration rates
    result = np.zeros([len(m_list), nday, sum(nwell), 2])
    for im in range(len(m_list)):
        m = m_list[im]
        result[im, :, :, :] = mig_rowseries_2dim(f2ivd, f2rsd, m, day=nday, well=nwell, distr_ini=None, inioutput=False)

    if save2file:
        with open(save2file, 'wb') as f:
            pickle.dump({'result': result, 'm_list': m_list, 'ini_abd': ini_abd, 'fin_abd': fin_abd}, f)
    return result

def sum_mod0community(path, tag, type, idx_list, outfile=None):
    """summarize resident communities/invaders
    Load all files f'{tag}_{idx}.pkl', extract key data and save as a single file f'sum_{tag}.pkl' or outfile
    idx are elements in idx_list
    
    :param path: path to load and save the files
    :param tag: tag for resident community/invader file names
    :param type: 'rsd' for resident communities, 'ivd' for invaders
    :param idx_list: list of indices of resident communities/invader
    :param outfile: file name to save the summarized data, if None, save as f'{path}sum_{tag}.pkl'
    """
    if type == 'rsd':
        var_list = ['mod_base', 'sol_base_daily', 'mod_rsd', 'sol_rsd_daily']
    elif type == 'ivd':
        var_list = ['mod_ivd', 'sol_ivd_daily']
    else:
        raise ValueError('type should be either rsd or ivd')

    nrsd = len(idx_list)
    sum_mod0 = np.empty(nrsd).astype('object')
    sum_mod0.fill(None)
    for idx in idx_list:
        try:
            with open(f'{path}{tag_rsd}_{idx}.pkl', 'rb') as f:
                data = pickle.load(f)
                data_part = {}
                for key in var_list:
                    data_part[key] = data[key]
                sum_mod0[idx] = copy.deepcopy(data_part)
        except FileNotFoundError:
            print(f'File not found: {tag_rsd}_{idx}.pkl')
    if outfile is None:
        outfile = f'{path}sum_{tag_rsd}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(sum_mod0, f)

def sum_mod1interaction(path, tag, rsd_list, ivd_list, outfile=None):
    """summarize interaction curves (also referred as competition curves)
    Load all files f'{tag}_ivd{iivd}rsd{irsd}.pkl', extract key data and save as a single file f'sum_{tag}.pkl' or outfile
    irsd are elements in rsd_list, iivd are elements in ivd_list

    :param path: path to load and save the files
    :param tag: tag for interaction curve file names
    :param rsd_list: list of indices of resident communities
    :param ivd_list: list of indices of invaders
    :param outfile: file name to save the summarized data, if None, save as f'{path}sum_{tag}.pkl
    """
    nrsd, nivd = len(rsd_list), len(ivd_list)
    sum_mod1comp = np.empty((nrsd, nivd)).astype('object')
    sum_mod1comp.fill(None)
    for irsd in rsd_list:
        for iivd in ivd_list:
            try:
                with open(f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl', 'rb') as f:
                    data = pickle.load(f)
                    sum_mod1comp[irsd, iivd] = copy.deepcopy(data)
            except FileNotFoundError:
                print(f'File not found: {tag}_ivd{iivd}rsd{irsd}.pkl')
                pass
    if outfile is None:
        outfile = f'{path}sum_{tag}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(sum_mod1comp, f)

def sum_mod2simulation(path, tag, rsd_list, ivd_list, outfile=None):
    """summarize simulated invasion results
    Load all files f'{tag}_rsd{irsd}ivd{iivd}.pkl', extract key data and save as a single file f'sum_{tag}.pkl' or outfile
    irsd are elements in rsd_list, iivd are elements in ivd_list

    :param path: path to load and save the files
    :param tag: tag for simulated invasion file names
    :param rsd_list: list of indices of resident communities
    :param ivd_list: list of indices of invaders
    :param outfile: file name to save the summarized data, if None, save as f'{path}sum_{tag}.pkl'
    """
    nrsd, nivd = len(rsd_list), len(ivd_list)
    sum_mod2invasion = np.empty((nrsd, nivd)).astype('object')
    sum_mod2invasion.fill(None)
    for irsd in rsd_list:
        for iivd in ivd_list:
            try:
                with open(f'{path}{tag}_rsd{irsd}ivd{iivd}.pkl', 'rb') as f:
                    data = pickle.load(f)
                    data_part = {}
                    for key in ['result', 'm_list']:
                        data_part[key] = data[key]
                    sum_mod2invasion[irsd, iivd] = copy.deepcopy(data_part)
                    print(f'Loaded {tag}_rsd{irsd}ivd{iivd}.pkl')
            except FileNotFoundError:
                print(f'File not found: {tag}_rsd{irsd}ivd{iivd}.pkl')
                pass
    if outfile is None:
        outfile = f'{path}sum_{tag}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(sum_mod2invasion, f)

def sum_mod3predinv(path, tag, rsd_list, ivd_list, outfile=None):
    """summarize predicted invasion results
    Load all files f'{tag}_ivd{iivd}rsd{irsd}.pkl', extract key data and save as a single file f'sum_{tag}.pkl' or outfile
    irsd are elements in rsd_list, iivd are elements in ivd_list

    :param path: path to load and save the files
    :param tag: tag for predicted invasion file names
    :param rsd_list: list of indices of resident communities
    :param ivd_list: list of indices of invaders
    :param outfile: file name to save the summarized data, if None, save as f'{path}sum_{tag}.pkl
    """
    nrsd, nivd = len(rsd_list), len(ivd_list)
    sum_mod3predinv = np.empty((nrsd, nivd)).astype('object')
    sum_mod3predinv.fill(None)
    for irsd in rsd_list:
        for iivd in ivd_list:
            try:
                with open(f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl', 'rb') as f:
                    data = pickle.load(f)
                    sum_mod3predinv[irsd, iivd] = copy.deepcopy(data)
            except FileNotFoundError:
                print(f'File not found: {tag}_ivd{iivd}rsd{irsd}.pkl')
                pass
    with open(f'{path}sum_{tag}.pkl', 'wb') as f:
        pickle.dump(sum_mod3predinv, f)


if __name__ == "__main__":
    #%%
    # =====Generate resident communities and invaders with CR model=====
    path = './cr/'
    tag_rsd = 'mod0rsd_20240212'  # output file tag for resident communities
    tag_ivd = 'mod0ivd_20240212'  # output file tag for invaders
    nrsd = 100
    nivd = 100
    # generate resident communities and invaders, simulate to equilibrium
    with ProcessPoolExecutor() as executor:
        for ii in range(nrsd):
            print(f'Rsd: Job distribution started for ii={ii}', flush=True)
            executor.submit(modgen_resident, ii, save2file=f'{path}{tag_rsd}_{ii}.pkl')
        for ii in range(nivd):
            print(f'Ivd: Job distribution started for ii={ii}', flush=True)
            executor.submit(modgen_invader, ii, save2file=f'{path}mod0ivd_20240212_{ii}.pkl')
    # summarize results - resident communities
    sum_mod0community(path, tag_rsd, 'rsd', range(nrsd))
    # summarize results - invaders
    sum_mod0community(path, tag_ivd, 'ivd', range(nivd))

    #%%
    # =====Simulate invasion with CR model=====
    path = './cr/'
    tag_rsd = 'mod0rsd_20240212'  # resident community file tag
    tag_ivd = 'mod0ivd_20240212'  # invader file tag
    tag = 'mod2invasion_20240212'  # output file tag for simulated invasion
    nrsd = 100
    nivd = 20
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    # simulate invasion for each pair of resident and invader models
    future_arr = np.empty((nrsd, nivd)).astype('object')
    with ProcessPoolExecutor(max_workers=60) as executor:
        for irsd in range(nrsd):
            print(f'Invasion: Job distribution started for irsd={irsd}', flush=True)
            for iivd in range(nivd):
                try:
                    # load resident and invader models
                    with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
                        data_rsd = pickle.load(f)
                        mod_rsd = data_rsd['mod_rsd']
                        NR_rsd = data_rsd['sol_rsd'].y[:, -1]
                    with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
                        data_ivd = pickle.load(f)
                        mod_ivd = data_ivd['mod_ivd']
                        NR_ivd = data_ivd['sol_ivd'].y[:, -1]
                except FileNotFoundError:
                    print(f'{tag_rsd}_{irsd}.pkl or {tag_ivd}_{iivd}.pkl not found in {path}, skiped')
                    continue
                # simulate invasion
                future_arr[irsd, iivd] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list,
                                                        save2file=f'{path}{tag}_rsd{irsd}ivd{iivd}.pkl')
        for future in as_completed(future_arr.flatten()):
            # error handling
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
    # summarize results
    sum_mod2simulation(path, tag, range(nrsd), range(nivd))

    #%%
    # =====Competition curve generation with CR model=====
    path = './cr/'
    tag_rsd = 'mod0rsd_20240212'  # resident community file tag
    tag_ivd = 'mod0ivd_20240212'  # invader file tag
    tag = 'mod1comp_20240215'  # output file tag for competition curve
    nrsd = 100
    nivd = 20
    # generate competition curve for each pair of resident and invader models
    mix_frac = np.array([0, 0.01, 0.1, 0.4, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]) / 100
    mix_frac = np.concatenate([mix_frac[:-1], 1 - mix_frac[::-1]])
    with ProcessPoolExecutor() as executor:
        for iivd in range(nivd):
            print(f'Competition: Job distribution started for iivd={iivd}', flush=True)
            for irsd in range(nrsd):
                try:
                    with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
                        data_rsd = pickle.load(f)
                        mod_rsd = data_rsd['mod_rsd']
                        NR_rsd = data_rsd['sol_rsd'].y[:, -1]
                    with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
                        data_ivd = pickle.load(f)
                        mod_ivd = data_ivd['mod_ivd']
                        NR_ivd = data_ivd['sol_ivd'].y[:, -1]
                except FileNotFoundError:
                    print(f'{tag_rsd}_{irsd}.pkl or {tag_ivd}_{iivd}.pkl not found in {path}, skiped')
                    continue
                executor.submit(comp_curve, mod_ivd, NR_ivd, mod_rsd, NR_rsd, mix_frac = mix_frac,
                                save2file=f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl')
    # summarize results
    sum_mod1interaction(path, tag, range(nrsd), range(nivd))

    #%%
    # =====Predict invasion from competition curve with CR model=====
    path = './cr/'
    tag_comp = 'mod1comp_20240215'  # competition curve file tag
    tag = 'mod3predinv_20240215'  # output file tag for predicted invasion
    nrsd = 100
    nivd = 20
    # make prediction for invasion from competition curve
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    with ProcessPoolExecutor() as executor:
        for irsd in range(nrsd):
            for iivd in range(nivd):
                try:
                    with open(f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl', 'rb') as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    print(f'{tag}_ivd{iivd}rsd{irsd}.pkl not found in {path}, skiped')
                    continue
                # ['mix_frac', 'ini_abd_allNR', 'fin_abd_allNR', 'mod_cmb', 'R_sup', 'df', 'randseed']
                mod_cmb = data['mod_cmb']
                ini_abd_allN = data['ini_abd_allNR'][:mod_cmb.nS, :]
                ini_abd_frac = ini_abd_allN[0, :] / ini_abd_allN.sum(axis=0)  # invader fraction
                fin_abd_allN = data['fin_abd_allNR'][:mod_cmb.nS, :]
                fin_abd_frac = fin_abd_allN[0, :] / fin_abd_allN.sum(axis=0)  # invader fraction
                df = data['df']
                ini_mix_frac = data['mix_frac']
                executor.submit(predinv_diff_m, ini_abd_frac, fin_abd_frac, ini_mix_frac, m_list,
                                save2file=f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl')
                # executor.submit(predinv_diff_m_2dim, ini_abd_allN, fin_abd_allN, m_list,
                #                 save2file=f'{path}mod3predinv2dim_20240215_ivd{iivd}rsd{irsd}.pkl')
    # summarize results
    sum_mod3predinv(path, tag, range(nrsd), range(nivd))

    #%%
    # =====Resident communities and invaders generation with LV model=====
    path = './lv/'
    tag_rsd = 'mod0rsd_20240216'  # output file tag for resident communities
    tag_ivd = 'mod0ivd_20240216'  # output file tag for invaders
    tag_pool = 'mod0lvmod_pool_20240216'  # output file tag for species pool with many resident and invader species
    nrsd_pool, nivd_pool = 1000, 100
    nrsd, nivd = 100, 20
    # generate resident communities and invaders, simulate to equilibrium
    lvmod_pool = BaseLotkaVolterraModel(nrsd_pool, nivd_pool, seed=2024)
    with open(f'{path}{tag_pool}.pkl', 'wb') as f:
        pickle.dump(lvmod_pool, f)
    future_arr = np.empty(nrsd + nivd).astype('object')
    with (ProcessPoolExecutor() as executor):
        for ii in range(nrsd):
            print(f'Rsd: Job distribution started for ii={ii}', flush=True)
            future_arr[ii] = executor.submit(lv_modgen_resident, lvmod_pool, ii,
                                             save2file=f'{path}{tag_rsd}_{ii}.pkl')
        for ii in range(nivd):
            print(f'Ivd: Job distribution started for ii={ii}', flush=True)
            future_arr[100 + ii] = executor.submit(lv_modgen_invader, lvmod_pool, ii,
                                                   save2file=f'{path}{tag_ivd}_{ii}.pkl')
        for future in as_completed(future_arr.flatten()):
            # error handling
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
    # summarize results - resident communities
    sum_mod0community(path, tag_rsd, 'rsd', range(nrsd))
    # summarize results - invaders
    sum_mod0community(path, tag_ivd, 'ivd', range(nivd))

    #%%
    # =====Interaction curve generation with LV model=====
    path = './lv/'
    tag_rsd = 'mod0rsd_20240216'  # resident community file tag
    tag_ivd = 'mod0ivd_20240216'  # invader file tag
    tag_pool = 'mod0lvmod_pool_20240216'  # species pool file tag
    tag = 'mod1comp_20240216'  # output file tag for competition curve
    # generate interaction curve for each pair of resident and invader models
    with open(f'{path}{tag_pool}.pkl', 'rb') as f:
        lvmod_pool = pickle.load(f)
    nrsd = 100
    nivd = 20
    mix_frac = np.array([0, 0.01, 0.1, 0.4, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]) / 100
    mix_frac = np.concatenate([mix_frac[:-1], 1 - mix_frac[::-1]])
    future_arr = np.empty((nrsd, nivd)).astype('object')
    with ProcessPoolExecutor() as executor:
        for iivd in range(nivd):
            print(f'Competition: Job distribution started for iivd={iivd}', flush=True)
            for irsd in range(nrsd):
                try:
                    with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
                        data_rsd = pickle.load(f)
                        mod_rsd = data_rsd['mod_rsd']
                        NR_rsd = data_rsd['sol_rsd'].y[:, -1]
                    with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
                        data_ivd = pickle.load(f)
                        mod_ivd = data_ivd['mod_ivd']
                        NR_ivd = data_ivd['sol_ivd'].y[:, -1]
                except FileNotFoundError:
                    print(f'{tag_rsd}_{irsd}.pkl or {tag_ivd}_{iivd}.pkl not found in {path}, skiped')
                    continue
                mod_cmb = mod_ivd.concat_community(mod_rsd, lvmod_pool)
                future_arr[irsd, iivd] = executor.submit(comp_curve, mod_ivd, NR_ivd, mod_rsd, NR_rsd,
                                                         mod_cmb=mod_cmb, mix_frac=mix_frac,
                                                         save2file=f'{path}{tag_comp}_rsd{irsd}ivd{iivd}.pkl')
        for future in as_completed(future_arr.flatten()):
            # error handling
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
    # summarize results
    sum_mod1interaction(path, tag, range(nrsd), range(nivd))

    #%%
    # =====Invasion prediction with LV model=====
    path = './lv/'
    tag_comp = 'mod1comp_20240216'  # competition curve file tag
    tag = 'mod3predinv_20240216'  # output file tag for predicted invasion
    nrsd = 100
    nivd = 20
    # make prediction for invasion from interaction curve
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    print(f'Predict invasion: Job distribution started', flush=True)
    with ProcessPoolExecutor() as executor:  # for small number of jobs, parallel processing is unnecessary, even slower
        for irsd in range(nrsd):
            for iivd in range(nivd):
                try:
                    with open(f'{path}{tag_comp}_ivd{iivd}rsd{irsd}.pkl', 'rb') as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    print(f'{tag_comp}_ivd{iivd}rsd{irsd}.pkl not found in {path}, skiped')
                    continue
                # ['mix_frac', 'ini_abd_allNR', 'fin_abd_allNR', 'mod_cmb', 'R_sup', 'df', 'randseed']
                mod_cmb = data['mod_cmb']
                ini_abd_allN = data['ini_abd_allNR'][:mod_cmb.nS, :]
                ini_abd_frac = ini_abd_allN[0, :] / ini_abd_allN.sum(axis=0)  # invader fraction
                fin_abd_allN = data['fin_abd_allNR'][:mod_cmb.nS, :]
                fin_abd_frac = fin_abd_allN[0, :] / fin_abd_allN.sum(axis=0)  # invader fraction
                df = data['df']
                ini_mix_frac = data['mix_frac']
                # parallel computation version
                executor.submit(predinv_diff_m, ini_abd_frac, fin_abd_frac, ini_mix_frac, m_list,
                                save2file=f'{path}{tag}_ivd{iivd}rsd{irsd}.pkl')
                # executor.submit(predinv_diff_m_2dim, ini_abd_allN, fin_abd_allN, m_list,
                #                 save2file=f'{path}mod3predinv2dim_20240216_ivd{iivd}rsd{irsd}.pkl')

                # # serial computation version
                # predinv_diff_m(ini_abd_frac, fin_abd_frac, ini_mix_frac, m_list,
                #                save2file=f'{path}mod3predinv_20240216_ivd{iivd}rsd{irsd}.pkl')
                # predinv_diff_m_2dim(ini_abd_allN, fin_abd_allN, m_list,
                #                     save2file=f'{path}mod3predinv2dim_20240216_ivd{iivd}rsd{irsd}.pkl')
    # summarize results
    sum_mod3predinv(path, tag, range(nrsd), range(nivd))

    #%%
    # =====Invasion simulation with LV model=====
    path = './lv/'
    tag_rsd = 'mod0rsd_20240216'  # resident community file tag
    tag_ivd = 'mod0ivd_20240216'  # invader file tag
    tag_pool = 'mod0lvmod_pool_20240216'  # species pool file tag
    tag_sim = 'mod2invasion_20240216'  # output file tag for simulated invasion
    nrsd = 100
    nivd = 20
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    # simulate invasion for each pair of resident and invader models
    with open(f'{path}{tag_pool}.pkl', 'rb') as f:
        lvmod_pool = pickle.load(f)
    future_arr = np.empty((nrsd, nivd)).astype('object')
    with ProcessPoolExecutor(max_workers=60) as executor:
        for irsd in range(nrsd):
            print(f'Invasion: Job distribution started for irsd={irsd}', flush=True)
            for iivd in range(nivd):
                try:
                    # load resident and invader models
                    with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
                        data_rsd = pickle.load(f)
                        mod_rsd = data_rsd['mod_rsd']
                        NR_rsd = data_rsd['sol_rsd'].y[:, -1]
                    with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
                        data_ivd = pickle.load(f)
                        mod_ivd = data_ivd['mod_ivd']
                        NR_ivd = data_ivd['sol_ivd'].y[:, -1]
                except FileNotFoundError:
                    print(f'{tag_rsd}_{irsd}.pkl or {tag_ivd}_{iivd}.pkl not found in {path}, skiped')
                    continue
                # simulate invasion
                mod_cmb = mod_ivd.concat_community(mod_rsd, lvmod_pool)
                future_arr[irsd, iivd] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd,
                                                         mod_cmb=mod_cmb, m_list=m_list,
                                                         save2file=f'{path}{tag_sim}_rsd{irsd}ivd{iivd}.pkl')
        for future in as_completed(future_arr.flatten()):
            # error handling
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
    # summarize results
    sum_mod2simulation(path, tag_sim, range(nrsd), range(nivd))

    #%%
    # =====Invasion prediction with LV model and 1-directional dispersal=====
    path = './lv/'
    tag_comp = 'mod1comp_20240802'  # competition curve file tag
    tag_pred = 'mod3predinv1dire_20240802'  # output file tag for predicted invasion
    nrsd = 20
    nivd = 20
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    # make prediction for invasion from interaction curve
    print(f'Predict invasion: Job distribution started', flush=True)
    with ProcessPoolExecutor() as executor:
        for irsd in range(nrsd):
            for iivd in range(nivd):
                try:
                    with open(f'{path}{tag_comp}_ivd{iivd}rsd{irsd}.pkl', 'rb') as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    print(f'{tag_comp}_ivd{iivd}rsd{irsd}.pkl not found in {path}, skiped')
                    continue
                # ['mix_frac', 'ini_abd_allNR', 'fin_abd_allNR', 'mod_cmb', 'R_sup', 'df', 'randseed']
                mod_cmb = data['mod_cmb']
                ini_abd_allN = data['ini_abd_allNR'][:mod_cmb.nS, :]
                ini_abd_frac = ini_abd_allN[0, :] / ini_abd_allN.sum(axis=0)  # invader fraction
                fin_abd_allN = data['fin_abd_allNR'][:mod_cmb.nS, :]
                fin_abd_frac = fin_abd_allN[0, :] / fin_abd_allN.sum(axis=0)  # invader fraction
                df = data['df']
                ini_mix_frac = data['mix_frac']
                executor.submit(predinv_diff_m, ini_abd_frac, fin_abd_frac, ini_mix_frac, m_list,
                                m_scalers=[0, 1], nwell=[2, 18],
                                save2file=f'{path}{tag_pred}_rsd{irsd}ivd{iivd}.pkl')
    # summarize results
    sum_mod3predinv(path, tag_pred, range(nrsd), range(nivd))

    #%%
    # =====Invasion simulation with LV model and 1-directional dispersal=====
    path = './lv/'
    tag_rsd = 'mod0rsd_20240216'  # resident community file tag
    tag_ivd = 'mod0ivd_20240216'  # invader file tag
    tag_pool = 'mod0lvmod_pool_20240216'  # species pool file tag
    tag_sim = 'mod2invasion1dire_20240802'  # output file tag for simulated invasion
    nrsd = 20
    nivd = 20
    m_list = np.array([0.2, 0.8, 2, 3.8, 6, 10, 20, 40]) / 100
    # simulate invasion for each pair of resident and invader models
    with open(f'{path}{tag_pool}.pkl', 'rb') as f:
        lvmod_pool = pickle.load(f)
    future_arr = np.empty((nrsd, nivd)).astype('object')
    with ProcessPoolExecutor(max_workers=60) as executor:
        for irsd in range(nrsd):
            print(f'Invasion: Job distribution started for irsd={irsd}', flush=True)
            for iivd in range(nivd):
                try:
                    # load resident and invader models
                    with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
                        data_rsd = pickle.load(f)
                        mod_rsd = data_rsd['mod_rsd']
                        NR_rsd = data_rsd['sol_rsd'].y[:, -1]
                    with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
                        data_ivd = pickle.load(f)
                        mod_ivd = data_ivd['mod_ivd']
                        NR_ivd = data_ivd['sol_ivd'].y[:, -1]
                except FileNotFoundError:
                    print(f'{tag_rsd}_{irsd}.pkl or {tag_ivd}_{iivd}.pkl not found in {path}, skiped')
                    continue
                # simulate invasion
                mod_cmb = mod_ivd.concat_community(mod_rsd, lvmod_pool)
                future_arr[irsd, iivd] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd,
                                                         mod_cmb=mod_cmb, m_list=m_list, m_scalers=[0, 1], nwell=[2, 18],
                                                         save2file=f'{path}{tag_sim}_rsd{irsd}ivd{iivd}.pkl')
        for future in as_completed(future_arr.flatten()):
            # error handling
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
    # summarize results
    sum_mod2simulation(path, tag_sim, range(nrsd), range(nivd))

    #%%
    # =====Invasion simulation with LV model and heterogeneous patch parameters=====
    path = './lv/'
    tag_rsd = 'mod0rsd_20240216'  # resident community file tag
    tag_ivd = 'mod0ivd_20240216'  # invader file tag
    tag_pool = 'mod0lvmod_pool_20240216'  # species pool file tag
    tag_sim = 'heterpatch_20240802'  # output file tag for simulated invasion
    # examples of pulsed invasions for testing the impact of patch heterogenity
    idx_explist = [[0, 4, 4],
                   [2, 3, 0],
                   [2, 7, 0],
                   [3, 5, 4],
                   [5, 5, 0],
                   [5, 13, 1],
                   [7, 18, 4],
                   [28, 6, 5]]
    # simulate invasion for each pair of resident and invader models
    for iexp in range(len(idx_explist)):
        irsd, iivd = idx_explist[iexp][:2]
        # Load example invader and resident communities
        with open(f'{path}{tag_ivd}_{iivd}.pkl', 'rb') as f:
            data_ivd = pickle.load(f)
            mod_ivd = data_ivd['mod_ivd']
            NR_ivd = data_ivd['sol_ivd_daily'][:, -1]
        with open(f'{path}{tag_rsd}_{irsd}.pkl', 'rb') as f:
            data_rsd = pickle.load(f)
            mod_rsd = data_rsd['mod_rsd']
            NR_rsd = data_rsd['sol_rsd_daily'][:, -1]
        # load mod_pool to combine resident and invader communities
        with open(f'{path}{tag_pool}.pkl', 'rb') as f:
            mod_pool = pickle.load(f)
        mod_cmb = mod_ivd.concat_community(mod_rsd, mod_pool)

        m_list = np.array([0.002, 0.008, 0.02, 0.038, 0.06, 0.1, 0.2, 0.4])

        noise_list = np.array([0, 0.05, 0.1, 0.2, 0.4])
        res_heter_m = np.zeros(len(noise_list), dtype='object')
        res_heter_K = np.zeros(len(noise_list), dtype='object')
        res_heter_r = np.zeros(len(noise_list), dtype='object')

        futures = np.empty((3, len(noise_list))).astype('object')
        with ProcessPoolExecutor(max_workers=5) as executor:
            for ii in range(len(noise_list)):
                futures[0, ii] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list, mod_cmb=mod_cmb,
                                                 R_sup=None, df=200, T=24, nday=20, nwell=[10, 10],
                                                 m_scalers=[1 / 2, 1 / 2], parallel_nworker=10, rand_m=noise_list[ii],
                                                 save2file=False, simple_out=False)
                futures[1, ii] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list, mod_cmb=mod_cmb,
                                                 R_sup=None, df=200, T=24, nday=20, nwell=[10, 10],
                                                 m_scalers=[1 / 2, 1 / 2], parallel_nworker=10, rand_K=noise_list[ii],
                                                 save2file=False, simple_out=False)
                futures[2, ii] = executor.submit(sim_invasion, mod_ivd, NR_ivd, mod_rsd, NR_rsd, m_list, mod_cmb=mod_cmb,
                                                 R_sup=None, df=200, T=24, nday=20, nwell=[10, 10],
                                                 m_scalers=[1 / 2, 1 / 2], parallel_nworker=10, rand_r=noise_list[ii],
                                                 save2file=False, simple_out=False)

        for ii in range(len(noise_list)):
            res_heter_m[ii] = futures[0, ii].result()
            res_heter_K[ii] = futures[1, ii].result()
            res_heter_r[ii] = futures[2, ii].result()

        # save to file
        data_rand = {}
        var_rand = ['res_heter_m', 'res_heter_K', 'res_heter_r', 'm_list', 'noise_list', 'mod_ivd', 'mod_rsd', 'mod_cmb']
        for var in var_rand:
            data_rand[var] = locals()[var]
        with open(f'{path}sum_{tag_sim}_rsd{irsd}_ivd{iivd}.pkl', 'wb') as f:
            pickle.dump(data_rand, f)