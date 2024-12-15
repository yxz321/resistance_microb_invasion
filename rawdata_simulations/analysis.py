import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os


def pos_above_thre(distr, f_thre, pos=None):
    """
    Position of invader fraction first above threshold for multiple distributions
    Interpolate between two adjacent positions to get a continuous position

    :param distr: np.array of shape (n_mig, n_day, n_well), fraction of invader
    :param f_thre: threshold for measuring invasion speed
    :param pos: np.array of shape (n_well), position of wells
    :return: np.array of shape (n_mig, n_day), position of invader fraction first above threshold
    """
    if len(distr.shape) == 1:
        distr = distr.reshape(1, -1)

    if pos is None:
        pos = np.arange(1, distr.shape[1] + 1)

    distr_reversed = np.flip(distr, axis=1)
    idxs = distr.shape[1] - 1 - np.argmax(distr_reversed > f_thre,
                                          axis=1)  # the last well position idx that's above threshold
    mask = np.max(distr > f_thre, axis=1)

    x_0 = pos[idxs]
    x_1 = np.roll(pos, -1)[idxs]

    f_0 = np.take_along_axis(distr, idxs[:, None], axis=1).flatten()
    f_1 = np.take_along_axis(np.roll(distr, -1, axis=1), idxs[:, None], axis=1).flatten()

    x = np.where(mask, (f_0 - f_thre) / (f_0 - f_1) * (x_1 - x_0) + x_0, np.nan)
    x[idxs == distr.shape[1] - 1] = np.nan

    return x


def get_rawdata_arr(file, sheet_name):
    """
    Read raw invasion experiment data.
    For a given sheet in combined raw data excel (e.g. SL_inv_OD.xlsx), read rawdata and return as nday*nrow*ncol array.

    :param file: str, path to the excel file
    :param sheet_name: str, name of the sheet to read
    :return: np.array of shape (nday, nrow, ncol), raw data
    """
    # Initialize variables
    data_blocks = []
    day_list = []

    # Iterate over the DataFrame in chunks of 9 rows at a time
    df_sn = pd.read_excel(file, sheet_name, header=None)
    for i in range(0, len(df_sn), 9):
        # Extract the day number from the first row of each block
        day = df_sn.iloc[i, 0]

        # Check if the day value is a string and follows the expected format
        if isinstance(day, str) and day.startswith("Day"):
            day_number = int(day.split(" ")[1])  # get the number after "Day"
            day_list.append(day_number)

            # Extract the data from the remaining rows of the block
            data_block = df_sn.iloc[i+1:i+9, :]
            data_blocks.append(data_block.values)
        else:
            print(f"Unexpected value in row {i}: {day}")

    # Convert the list of data blocks to a 3D numpy array
    return np.array(data_blocks).astype('float64'), np.array(day_list)  # nday * 8 rows * 12 cols


def remove_scatter(data_raw, s_d=0.08730987, s_u=0.02029704, s_r=0.05823732, s_l=0.03478871, bg=20):
    """
    remove scattered light from luciferin readout. parameters calibrated for the black plates

    :param x: raw luciferin readout, layout in the shape of the plate
    :param bg: background, i.e. luciferin readout for blank control wells
    """
    x = data_raw - bg
    data = x * 1  # make a deep copy

    data[:-1] -= s_d * (x[1:])  # scattering from lower well
    data[1:] -= s_u * (x[:-1])  # scattering from up
    data[:, :-1] -= s_r * (x[:, 1:])  # scattering from right
    data[:, 1:] -= s_l * (x[:, :-1])  # scattering from left
    data += bg

    return data.astype('int32')


def calc_inv_speed(series_allm, f_thre):
    """
    Calculate invasion speed for multiple distributions

    :param series_allm: np.array of shape (n_mig, n_day, n_well), fraction of invader
    :param f_thre: threshold for measuring invasion speed
    :return: np.array of shape (n_mig, n_day), instantaneous invasion speed
    """
    # if series_allm just has one migration rate, reshape it to check it is 3D
    if len(series_allm.shape) == 2:
        series_allm = series_allm.reshape(1, *series_allm.shape)
    vn_allm = np.zeros(series_allm.shape[:2])
    vn_allm[:, :] = np.nan
    for im in range(series_allm.shape[0]):
        series = series_allm[im, :]
        pos = pos_above_thre(series, f_thre)
        vn = pos[1:] - pos[:-1]
        vn_allm[im, 1:] = vn
    return vn_allm


def draw_wave_speeds(wave_speeds, time_points=None, ax=None,
                     exclude_from_mean=1, vmax=1, vmin=0, datamax=1.5, datamin=-0.5, cbar=False):
    """
    Draws wave speed data on a given axis, with options to adjust the visualization.

    Parameters:
    - wave_speeds (array-like): 1-D Array of wave speeds to plot.
    - time_points (array-like, optional): Array of time points corresponding to the wave speeds. Defaults to a range starting from 1.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - exclude_from_mean (int, optional): Index from which to start calculating the mean for color mapping. Defaults to 1.
    - vmax (float, optional): The maximum value for normalizing the colormap. Defaults to 1.
    - vmin (float, optional): The minimum value for normalizing the colormap. Defaults to 0.
    - datamax (float, optional): The maximum y-axis limit for the plot. Defaults to 1.5.
    - datamin (float, optional): The minimum y-axis limit for the plot. Defaults to -0.5.

    Returns:
    None: This function does not return any value. It only modifies the ax passed or creates a new plot.
    """
    if ax is None:
        fig = plt.figure(figsize=[4.8, 3])
        ax = plt.gca()
    if time_points is None:
        time_points = np.arange(len(wave_speeds)) + 1

    mean = np.nanmean(wave_speeds[exclude_from_mean:])

    # Get the colormap
    cmap = mpl.colormaps['Oranges']
    # Normalize the value based on vmin and vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Get the color from the colormap
    color = cmap(norm(mean))
    # Set the face color of the axis
    ax.set_facecolor(color)

    # Overlay the raw time series
    ax.plot(time_points, wave_speeds,
            color=[0.2, 0.2, 0.2], path_effects=[pe.Stroke(linewidth=8, foreground='white'), pe.Normal()])
    ax.set_ylim([datamin, datamax])

    if cbar:
        axins = ax.inset_axes([1.05, 0, 0.05, 1])
        hcbar = plt.gcf().colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),
                                   cax=axins, label='Mean invasion speed')
        plt.sca(ax)
        return hcbar


def sum_vn(nrsd, nivd, file_tag, file_path='', thre=0.5, save=True):
    """
    Summarize invasion speed for multiple distributions.
    Different from the sum* functions in main_simulations.py, the output is much smaller
    Can handle both simulation (mod2invasion) and prediction (mod3predinv) data.

    :param nrsd: number of resident communities
    :param nivd: number of invaders
    :param file_tag: tag for the files, file name should be like f'{file_tag}_rsd{irsd}ivd{iivd}.pkl'
    :param file_path: path to the files
    :param thre: invader fraction threshold for measuring invasion speed. pass to calc_inv_speed
    :return: m_list, sum_vn
        sum_vn: np.array of shape (nrsd, nivd, nmig, nday), invasion speed for multiple scenarios
    """
    firstfile = True  # flag for first file
    for irsd in range(nrsd):
        print(f'Loading rsd{irsd}...')
        for iivd in range(nivd):
            if f'{file_tag}_rsd{irsd}ivd{iivd}.pkl' in os.listdir(file_path):
                file_name = f'{file_tag}_rsd{irsd}ivd{iivd}.pkl'
            elif f'{file_tag}_ivd{iivd}rsd{irsd}.pkl' in os.listdir(file_path):
                file_name = f'{file_tag}_ivd{iivd}rsd{irsd}.pkl'
            else:
                print(f'No file found for rsd{irsd}ivd{iivd}, skiped')
                continue

            try:
                with open(f'{file_path}{file_name}', 'rb') as f:
                    data = pickle.load(f)
                # the result is stored in different keys in simulation and prediction data
                if 'res' in data.keys():
                    key = 'res'
                elif 'result' in data.keys():
                    key = 'result'
                else:
                    raise KeyError('No key "res" or "result" found')
                res = data[key]

                # initialize sum_vn when first file is loaded
                if firstfile:
                    m_list = data['m_list']
                    nmig = res.shape[0]
                    nday = res.shape[1]
                    nwell = res.shape[2]
                    sum_vn = np.zeros((nrsd, nivd, nmig, nday))
                    sum_vn.fill(np.nan)
                    firstfile = False

                # if 'invmod' in data.keys():  # simulation data
                #     invmod = data['invmod']
                #     series_allm = res[:, :, :, invmod.nS - 1] / res[:, :, :, :invmod.nS].sum(axis=-1)
                if 'mod_rsd' in data.keys():  # simulation data
                    mod_rsd, mod_ivd = data['mod_rsd'], data['mod_ivd']
                    series_allm = res[:, :, :, 0] / res[:, :, :, :mod_rsd.nS + mod_ivd.nS].sum(axis=-1)
                else:  # prediction data
                    series_allm = res
                vn_allm = calc_inv_speed(series_allm, f_thre=thre)
                sum_vn[irsd, iivd, :, :] = vn_allm
            except FileNotFoundError:
                print(f'{file_path}{file_name} not found in {file_path}, skiped')
                continue

    if save:
        if firstfile is False:
            with open(f'{file_path}sum_vn_{file_tag}_thre={thre}.pkl', 'wb') as f:
                pickle.dump({'sum_vn': sum_vn, 'm_list': m_list}, f)
        else:
            print('No file found, no sum_vn saved')

    return m_list, sum_vn


if __name__ == "__main__":
    # # Summarizing invasion speeds for Lotka-Volterra model
    # path = './lv/'
    # # calculate and save invasion speed for simulation data
    # sum_vn(100, 20, file_tag='mod2invasion_20240216', file_path=path)
    # # calculate and save invasion speed for prediction data
    # sum_vn(100, 20, file_tag='mod3predinv_20240216', file_path=path)
    #
    # # Summarizing invasion speeds for Consumer-Resource model
    # path = './cr/'
    # sum_vn(100, 20, file_tag='mod2invasion_20240212', file_path=path)
    # sum_vn(100, 20, file_tag='mod3predinv_20240215', file_path=path)

    # Summarizing invasion speeds for Lotka-Volterra model with 1-directional dispersal
    path = './lv/'
    for thre in [0.4, 0.5, 0.6]:
        sum_vn(20, 20, file_tag='mod3predinv1dire_20240802', file_path=path, thre=thre)
        sum_vn(20, 20, file_tag='mod2invasion1dire_20240802', file_path=path, thre=thre)


