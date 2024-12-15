import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def frac_change_func(thre=0.3, steep=10):
    """
    Given thre and steep, return a function handle that encodes a S-shaped function.
    :param thre: position of middle point
    :param steep: steepness of the function
    :return: func: the function handle
    """
    def func_erf_cdf(f0):
        """
        The final fraction of Su after 1d culture as a function of the initial fraction of Su.
        :param f0: initial fraction of Su
        :param thre: f(thre) = 1/2
        :param steep: the higher the steeper the step
        :return f: the final fraction of Su after 1d culture
        """
        f = erf((f0 - thre) * steep)
        # normalize to [0, 1]
        f_max = erf((1 - thre) * steep)
        f_min = erf((0 - thre) * steep)
        f = (f - f_min) / (f_max - f_min)
        return f
    return func_erf_cdf


def mig_rowseries(g, m, day=10, well=[6, 6], distr_ini = None,
                  draw=False, inioutput=False):
    """
    Generate a time series of modeled migration scheme, each entry is a list of species composition (range [0, 1],
    0 for Lp, 1 for Su) of each well at the corresponding time
    :param g: function handle f=g(f0), where f is the final fraction of Su after 1d culture, and f0 is the initial
    fraction of Su
    :param m: migration rate
    :param day: number of days to simulate
    :param well: a listlike of length 2, the starting number of wells for [Su, Lp]
    :param draw: if True directly make the wave propagation plot for single well
    :param inioutput: if True also return iniseries, which records initial communities at start of each day
    :return finalseries: a list timeseries, each entry is a np.array representing the species composition of wells
    (corresponding to a row in experiment) of the timepoint
    :return iniseries: a list of timeseries, each entry is a list of dictionaries {"dilu", "left", "right"}
    dilu: initial community from dilution leftover of old well
    left: initial community from migration from the left well
    right: initial community from migration from the right well
    """
    finalseries = []
    iniseries = []
    # initial species distribution
    if distr_ini:
        distr = distr_ini
    else:
        distr = np.array([1] * well[0] + [0] * well[1])

    for d in range(day):
        finalseries.append(distr)

        ini_dilu = distr * (1-m)  # initial community from dilution leftover of old well
        ini_left = np.concatenate(([0], distr[:-1])) * m/2  # initial community from migration from the left well
        ini_right = np.concatenate((distr[1:], [0])) * m/2  # initial community from migration from the right well
        # add together and rescale border wells
        ini_distr = (ini_dilu + ini_left + ini_right) / np.concatenate(([1-m/2], [1] * (sum(well) - 2), [1-m/2]))
        ini_distr = np.array([round(z, 15) for z in ini_distr])  # discard the last two digit (not precise) of float64
        if inioutput:
            iniseries.append({"dilu":ini_dilu, "left":ini_left, "right":ini_right})

        distr = g(ini_distr)  # final community after growth (modelled as fraction change of species composition)

    if draw:
        colors = np.linspace(1, 0, day)
        for d in range(day):
            plt.plot(range(1, sum(well)+1), finalseries[d], color=colors[d] * np.array([1, 1, 1]))

    if inioutput:
        return finalseries, iniseries
    else:
        return finalseries


def mig_rowseries_v2(g, m, day=10, well=[6, 6], distr_ini=None, m_scalers=[1 / 2, 1 / 2],
                     draw=False, inioutput=False):
    """
    Generate a time series of modeled migration scheme, each entry is a list of species composition (range [0, 1],
    0 for Lp, 1 for Su) of each well at the corresponding time
    :param g: function handle f=g(f0), where f is the final fraction of Su after 1d culture, and f0 is the initial
    fraction of Su
    :param m: migration rate
    :param day: number of days to simulate
    :param well: a listlike of length 2, the starting number of wells for [Su, Lp]
    :param m_scalers: scaling factors [left, right] for m, e.g. [1/2, 1/2] for symmetric dispersal,
        [0, 1] for one-way dispersal from left to right
    :param draw: if True directly make the wave propagation plot for single well
    :param inioutput: if True also return iniseries, which records initial communities at start of each day
    :return finalseries: a list timeseries, each entry is a np.array representing the species composition of wells
    (corresponding to a row in experiment) of the timepoint
    :return iniseries: a list of timeseries, each entry is a list of dictionaries {"dilu", "left", "right"}
    dilu: initial community from dilution leftover of old well
    left: initial community from migration from the left well
    right: initial community from migration from the right well
    """
    m_l, m_r = m * np.array(m_scalers)

    finalseries = np.zeros((day, well[0] + well[1]))
    iniseries = []
    # initial species distribution
    if distr_ini:
        distr = distr_ini
    else:
        distr = np.array([1] * well[0] + [0] * well[1])

    for d in range(day):
        finalseries[d, :] = distr

        ini_distr = np.zeros(distr.shape)
        ini_distr[0] = distr[0] * (1 - m_l) + distr[1] * m_l  # leftmost well
        ini_distr[-1] = distr[-1] * (1 - m_r) + distr[-2] * m_r  # rightmost well
        ini_distr[1:-1] = distr[1:-1] * (1 - m_l - m_r) + distr[:-2] * m_r + distr[2:] * m_l  # other wells
        # discard the last two digit (not precise) of float64. This avoids numerical error driven <0 or >1 values.
        ini_distr = np.round(ini_distr, 15)
        if inioutput:
            iniseries.append({"distr": distr, "ini_distr": ini_distr})

        distr = g(ini_distr)  # final community after growth (modelled as fraction change of species composition)

    if draw:
        colors = np.linspace(1, 0, day)
        for d in range(day):
            plt.plot(range(1, sum(well) + 1), finalseries[d, :], color=colors[d] * np.array([1, 1, 1]))

    if inioutput:
        return finalseries, iniseries
    else:
        return finalseries
# def mig_rowseries_v2_old(g, m, day=10, well=[6, 6], distr_ini = None, m_scalers=[1/2, 1/2],
#                   draw=False, inioutput=False):
#     """
#     Generate a time series of modeled migration scheme, each entry is a list of species composition (range [0, 1],
#     0 for Lp, 1 for Su) of each well at the corresponding time
#     :param g: function handle f=g(f0), where f is the final fraction of Su after 1d culture, and f0 is the initial
#     fraction of Su
#     :param m: migration rate
#     :param day: number of days to simulate
#     :param well: a listlike of length 2, the starting number of wells for [Su, Lp]
#     :param m_scalers: scaling factors [left, right] for m, e.g. [1/2, 1/2] for symmetric dispersal,
#         [0, 1] for one-way dispersal from left to right
#     :param draw: if True directly make the wave propagation plot for single well
#     :param inioutput: if True also return iniseries, which records initial communities at start of each day
#     :return finalseries: a list timeseries, each entry is a np.array representing the species composition of wells
#     (corresponding to a row in experiment) of the timepoint
#     :return iniseries: a list of timeseries, each entry is a list of dictionaries {"dilu", "left", "right"}
#     dilu: initial community from dilution leftover of old well
#     left: initial community from migration from the left well
#     right: initial community from migration from the right well
#     """
#     finalseries = np.array((day, well[0] + well[1]))
#     iniseries = []
#     # initial species distribution
#     if distr_ini:
#         distr = distr_ini
#     else:
#         distr = np.array([1] * well[0] + [0] * well[1])
#
#     for d in range(day):
#         finalseries[d, :] = distr
#
#         ini_dilu = distr * (1-m)  # initial community from dilution leftover of old well
#         ini_left = np.concatenate(([0], distr[:-1])) * m/2  # initial community from migration from the left well
#         ini_right = np.concatenate((distr[1:], [0])) * m/2  # initial community from migration from the right well
#         # add together and rescale border wells
#         ini_distr = (ini_dilu + ini_left + ini_right) / np.concatenate(([1-m/2], [1] * (sum(well) - 2), [1-m/2]))
#         ini_distr = np.array([round(z, 15) for z in ini_distr])  # discard the last two digit (not precise) of float64
#         if inioutput:
#             iniseries.append({"dilu":ini_dilu, "left":ini_left, "right":ini_right})
#
#         distr = g(ini_distr)  # final community after growth (modelled as fraction change of species composition)
#
#     if draw:
#         colors = np.linspace(1, 0, day)
#         for d in range(day):
#             plt.plot(range(1, sum(well)+1), finalseries[d, :], color=colors[d] * np.array([1, 1, 1]))
#
#     if inioutput:
#         return finalseries, iniseries
#     else:
#         return finalseries


def mig_rowseries_2dim(g_ivd, g_rsd, m, day=10, well=[6, 6], distr_ini=None, inioutput=False):
    """
    Generate a time series of modeled migration scheme, based on 2-dimensional interaction curves
    :param g_ivd: function handle f=g_ivd(f0), where f is the final fraction of invader after 1 day culture,
        and f0 is the initial fraction of invader
    :param g_rsd: function handle f=g_rsd(f0), where f is the final fraction of resident after 1 day culture,
        and f0 is the initial fraction of resident
    :param m: migration rate
    :param day: number of days to simulate
    :param well: a listlike of length 2, the starting number of wells for [invader, resident]
    :return finalseries: nday * sum(well) * 2, timeseries of modelled row species composition
    """

    # initial species distribution
    if distr_ini:
        distr = distr_ini
    else:
        distr = np.zeros((2, sum(well)))
        distr[0, :well[0]] = g_ivd(1)  # invader
        distr[1, well[0]:] = g_rsd(0)  # resident
    iniseries = np.zeros((day, sum(well), 2))
    iniseries.fill(np.nan)
    # final species distribution
    finalseries = np.zeros((day, sum(well), 2))
    finalseries.fill(np.nan)

    for d in range(day):
        finalseries[d, :, :] = distr.T

        # calculate initial distribution
        ini_dilu = distr * (1 - m)  # initial community from dilution leftover of old well
        ini_left = np.concatenate((distr[:, 0][:, np.newaxis], distr[:, :-1]), axis=1) * m/2  # initial community from migration from the left well
        ini_right = np.concatenate((distr[:, 1:], distr[:, -1][:, np.newaxis]), axis=1) * m/2  # initial community from migration from the right well
        # add together
        ini_distr = ini_dilu + ini_left + ini_right
        ini_distr = np.round(ini_distr, 15)  # discard the last two digit (not precise) of float64
        iniseries[d, :, :] = ini_distr.T  # note down initial distribution

        # calculate final distribution
        ini_distr_frac = ini_distr[0, :] / np.sum(ini_distr, axis=0)  # calculate initial fraction
        distr = np.zeros((2, sum(well)))
        distr[0, :] = g_ivd(ini_distr_frac)  # final invader abundance distribution
        distr[1, :] = g_rsd(ini_distr_frac)  # final resident abundance distribution

    if inioutput:
        return finalseries, iniseries
    else:
        return finalseries

    
def mig_plateseries(g, day=10, well=[6, 6], distr_ini=None,
                    mig_rates=np.array([0.02, 0.1, 0.2, 0.6, 1.2, 2, 6, 10]) / 20):
    """
    generate timeseries of modelled migration plates (directly correspond to experimental measurement)
    :param g: function handle of fraction change function, or a listlike of 8 function handles corresponding to each row
    :param day: number of days to simulate
    :param well: a listlike of length 2, the starting number of wells for [Su, Lp]
    :param mig_rates: a listlike of length 8, migration rates that is used for each row
    :return: timeseries of pandas.Dataframes, each entry represent plate species composition of that day (0:Lp, 1:Su)
    """
    plateseries = [pd.DataFrame()] * day

    listlike = (list, np.ndarray, tuple)
    # if g and mig_rates are listlikes, check whether they have same length
    # if one/both are single values, convert it to list
    if isinstance(g, listlike):
        if isinstance(mig_rates, listlike):
            if len(g) != len(mig_rates):
                raise ValueError(
                    "g and mig_rates must be single value or listlike of same length (number of modelled rows)")
        else:
            mig_rates = [mig_rates] * len(g)
    else:
        if isinstance(mig_rates, listlike):
            g = [g] * len(mig_rates)
        else:
            g = [g]
            mig_rates = [mig_rates]
    # check if well is a listlike of length 2
    if not isinstance(well, listlike) or len(well) != 2:
        raise ValueError("well must be a listlike of length 2")

    for row, m in enumerate(mig_rates):
        series_newrow = mig_rowseries(g[row], m=m, day=day, well=well, distr_ini=distr_ini)  # generate time series for the current row
        for d in range(day):  # append the row time series to modelled plate of each day
            temp_df = plateseries[d]
            plateseries[d] = temp_df.append(pd.DataFrame([series_newrow[d]]))

    return plateseries


mList = np.array([0.02, 0.1, 0.2, 0.6, 1.2, 2, 6, 10]) / 20
if __name__ == "__main__":
    threshold = 0.2
    steepness = 10

    # fraction change function
    func = frac_change_func(threshold, steepness)

    plt.plot(np.linspace(0, 1, 1000), func(np.linspace(0, 1, 1000)), )
