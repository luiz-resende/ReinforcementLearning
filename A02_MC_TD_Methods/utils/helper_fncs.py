###############################################################################
#  Copyright (c) 2021 by Luiz Resende Silva
#  Released under the GNU General Public License; see LICENSE.md for details.
################################################################################
#  The functions below are constructed based on the environment Modified Frozen
#  Lake by Michel Ma
#  [source code](https://github.com/micklethepickle/modified-frozen-lake)
################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tqdm
import copy
import urllib


def import_github_script(URL=r'https://raw.githubusercontent.com/micklethepickle/modified-frozen-lake/main/frozen_lake.py',
                         file_name='frozen_lake.py'):
    """
    Function retrieves python script from raw github and created an importable file with extension '.py'.

    Parameters
    ----------
    URL : str, optional
        URL address containing the file. The default is
        r'https://raw.githubusercontent.com/micklethepickle/modified-frozen-lake/main/frozen_lake.py'.
    file_name : str, optional
        Name to be given to the downloaded file. The default is 'frozen_lake.py'.

    Returns
    -------
    None.
    """
    # path = path + file_name
    path = file_name
    if (not os.path.isfile(path)):
        raw_bytes = urllib.request.urlopen(URL).read()
        raw_str = raw_bytes.decode("utf-8")
        mode = 'x'
        text_file = open(path, mode)
        text_file.write(raw_str)
        text_file.close()


def plotter_mult(xy_data, x_label="X Axis", y_label="Y Axis", graph_title="My Plot", use_x_limits=False, x_limits=(0, 100),
                 use_y_limits=False, y_limits=(0, 100), use_x_ticks=False, x_ticks=1, use_y_ticks=False, y_ticks=1,
                 line_types=[], line_size=2, color_list=[], plot_legend=True, legend_loc='best', number_legend_cols=1,
                 legend_font_size=1.0, use_log_scale_x=False, use_log_scale_y=False, figure_size=[9, 6],
                 title_font_size=26, save_plot=False, save_directory="same"):
    """
    Function graphically represents the results of f(x) (up to 28 different functions) for a set of values x=(1,...,n).

    Parameters
    ----------
    xy_data : pandas.DataFrame
        Structure with labelled xy data points, where first column contains the x points and the subsequent columns the different f(x) values.
    x_label : str, optional
        Label for the x axis. The default is "X Axis".
    y_label : str, optional
        Label for the y axis. The default is "Y Axis".
    graph_title : str, optional
        Title for the plot. The default is "My Plot".
    use_x_limits : bool, optional
        Boolean flag to whether or not use preset minimum and maximum values for x axis. The default is False.
    x_limits : tuple, optional
        Tuple with the minimum and maximum limit values for the x axis. The default is (0, 100).
    use_y_limits : bool, optional
        Boolean flag to whether or not use preset minimum and maximum values for y axis. The default is False.
    y_limits : tuple, optional
        Tuple with the minimum and maximum limit values for the y axis. The default is (0, 100).
    use_x_ticks : bool, optional
        Boolean flag to whether or not use predefined minimum x axis increment unit. The default is False.
    x_ticks : int, optional
        X axis increment unit. The default is 1.
    use_y_ticks : bool, optional
        Boolean flag to whether or not use predefined minimum y axis increment unit. The default is False.
    y_ticks : int, optional
        Y axis increment unit. The default is 1.
    line_types : list, optional
        List with strings defining line types, e.g. ['-', '--', '.-']. The default is [].
    line_size : int, optional
        The integer number for line width. The default is 2.
    color_list : list, optional
        The list of strings for the colors, e.g. ['red', 'blue', 'green']. The default is [].
    plot_legend : bool, optional
        Flag to whether or not include the legend to the plot. The default is True.
    legend_loc : str, optional
        String for the legend location in the plot, e.g. {'best', 'upper right', 'upper left', 'lower left', 'lower right',
        'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}. The default is 'best'.
    number_legend_cols : int, optional
        Number of columns to divide legend. The default is 1.
    legend_font_size : float, optional
        Percentage of title_font_size to use as font size for legend. The default is 1.0.
    use_log_scale : bool, optional
        Boolean flag to whether or not use logarithm scale for the x axis. The default is False.
    figure_size : list, optional
        List with two int values representing figure size [width, hight]. The default is [15, 10].
    title_font_size : int, optional
        Title font size, which is also used for the axes title sizes (decreased by 4 units). The default is 26.
    save_plot : bool, optional
        Boolean flag to whether or not save the plot as a PNG file. The default is False.
    save_directory : str, optional
        Either "same" for saving in the same folder as the script or "C:\\...\\...\\..." directory where to save figures. The default is "".

    Raises
    ------
    TypeError
        Argument 'xy_data' passed is not a pandas.DataFrame!
    ValueError
        Argument 'xy_data' passed has more then 28 f(x) columns...
    """
    try:
        if (not isinstance(xy_data, pd.DataFrame)):
            raise TypeError("TypeError: Argument 'xy_data' passed is not a pandas.DataFrame!")
        columns = list(xy_data.columns.values)
        if (color_list == 0 or color_list == 1 or color_list == []):
            color_list = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600',
                          '#89023e', '#932829', '#924618', '#89610f', '#797820', '#648d40', '#47a069', '#15b097']
        elif (color_list == 2):
            color_list = ['#0e0e0e', '#797575', '#9e8eff', '#d74d74', '#242b89', '#4b6bc8', '#6e00bb', '#ba00ff',
                          '#65442e', '#992828', '#da6201', '#ffc11b', '#168f77', '#7ecfce', '#6ba23e', '#bdd900']
        else:
            color_list = ['red', 'green', 'blue', 'orange', 'yellow', 'magenta', 'black', 'cyan']
        if (len(line_types) == 0):
            line_types = ['-', '-.', ':', '--']
        if ((len(columns) - 1) > (len(color_list) * len(line_types))):
            raise ValueError("ValueError: Argument 'xy_data' has more then %d f(x) columns. Increase unique colors or line types."
                             % (len(color_list) * len(line_types)))
        fig, ax = plt.subplots(figsize=figure_size)
        ax.grid(True)
        if (use_log_scale_x is True):
            ax.set_xscale('log')
        if (use_log_scale_y is True):
            ax.set_yscale('log')
        if (len(columns) <= (len(color_list) + 1)):
            for i in range(len(columns) - 1):
                plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[0], markersize=None,
                         color=color_list[i], linewidth=line_size)
        else:
            cc = 0
            lt = 0
            for i in range(len(columns) - 1):
                plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[lt],
                         markerfacecolor=None, markersize=None, color=color_list[cc], linewidth=line_size)
                cc += 1
                if (cc == len(color_list)):
                    lt += 1
                    cc = 0
                if (lt == len(line_types)):
                    break
        if (graph_title != ''):
            plt.suptitle(graph_title, fontsize=title_font_size)
        if(x_label == "X Axis"):
            x_label = columns[0]
        if (use_log_scale_x):
            x_label = x_label + " (log scale)"
        if (use_log_scale_y):
            y_label = y_label + " (log scale)"
        plt.xlabel(x_label, fontsize=(title_font_size - 4))
        plt.ylabel(y_label, fontsize=(title_font_size - 4))
        if (use_x_limits):
            plt.xlim(x_limits[0], x_limits[1])
            if(use_x_ticks):
                ax.xaxis.set_ticks(np.arange(x_limits[0], x_limits[1], x_ticks))
        elif (use_x_ticks):
            ax.xaxis.set_ticks(np.arange(xy_data[columns[0]].min(), xy_data[columns[0]].max(), x_ticks))
        if (use_y_limits):
            plt.ylim(y_limits[0], y_limits[1])
            if (use_y_ticks):
                ax.yaxis.set_ticks(np.arange(y_limits[0], y_limits[1], y_ticks))
        elif (use_y_ticks):
            ax.yaxis.set_ticks(np.arange(xy_data.iloc[:, 1:].min().min(), xy_data.iloc[:, 1:].max().max(), y_ticks))
        if (plot_legend):
            plt.legend(loc=legend_loc, fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=number_legend_cols,
                       fontsize=int(title_font_size * legend_font_size))
        plt.show()
        if (save_plot):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            file_name = graph_title.replace('\\', "").replace('$', "").replace(' ', "").replace('vs.', "").replace(':', "_")
            if (save_directory == "same"):
                plt.savefig(timestr + file_name + ".png")
            else:
                plt.savefig(save_directory + "/" + timestr + file_name + ".png")
    except Exception as e:
        if (type(e) == TypeError or type(e) == ValueError):
            print(e)
        else:
            print("UnknownError: Problem while running 'plotter_mult()'. Please, review arguments passed...")
            print("ERROR MESSAGE: %s" % (e))


def plotter_mean(data_xy, num_datasets=1, x_label="X Axis", y_label="Y Axis", graph_title="My Plot", root_name='newcolumn', plot_std_dev=True,
                 alpha=0.2, use_x_limits=False, x_limits=(0, 100), use_y_limits=False, y_limits=(0, 100),
                 use_x_ticks=False, x_ticks=1, use_y_ticks=False, y_ticks=1, colors='#003f5c', marker_type='', mark_size=None,
                 line_type='-', line_size=2, plot_legend=True, legend_loc='best', legend_font_size=1.0, use_log_scale=False,
                 figure_size=[9, 6], title_font_size=20, save_plot=False, save_directory="same"):
    """
    Function takes as input a pandas.DataFrame containing different f(x), calculates their mean and standard deviation and
    plots the results.

    Parameters
    ----------
    data_xy : pandas.DataFrame
        Structure with labelled xy data points, where first column contains the x points and the subsequent columns the different f(x) values.
    num_datasets : int, optional
        The number of different datasets to calculated the mean and plot. The default is 1.
    x_label : str, optional
        Label for the x axis. The default is "X Axis".
    y_label : str, optional
        Label for the y axis. The default is "Y Axis".
    graph_title : str, optional
        Title for the plot. The default is "My Plot".
    root_name : str, optional
        String with root name for new columns. The default is 'newcolumn'.
    plot_std_dev : bool, optional
        Flag to whether or not plot standard deviation, The default is True.
    alpha : float, optional
        Percentage of 'seethrough' on standard dev, color. The default is 0.2.
    use_x_limits : bool, optional
        Boolean flag to whether or not use preset minimum and maximum values for x axis. The default is False.
    x_limits : tuple, optional
        Tuple with the minimum and maximum limit values for the x axis. The default is (0, 100).
    use_y_limits : bool, optional
        Boolean flag to whether or not use preset minimum and maximum values for y axis. The default is False.
    y_limits : tuple, optional
        Tuple with the minimum and maximum limit values for the y axis. The default is (0, 100).
    use_x_ticks : bool, optional
        Boolean flag to whether or not use predefined minimum x axis increment unit. The default is False.
    x_ticks : int, optional
        X axis increment unit. The default is 1.
    use_y_ticks : bool, optional
        Boolean flag to whether or not use predefined minimum y axis increment unit. The default is False.
    y_ticks : int, optional
        Y axis increment unit. The default is 1.
    colors : str, optional
        String for the color to be used, e.g. {'red', 'green', 'blue', 'yellow', 'magenta', 'black', 'cyan'}. The default is 'red'.
    marker_type : str, optional
        String for the marker type, e.g. {'', '.', 'o', 'v', 'p', 'D', 's', '+'}. The default is '' (no marker).
    mark_size : int, optional
        Size of the marker if marker defined. The default is None.
    line_type : str, optional
        String for the line type to be used, e.g. {'-', '-.', ':', '--'}. The default is '-' (solid line).
    line_size : int, optional
        Line width to be used. The default is 2.
    plot_legend : bool, optional
        Flag to whether or not include the legend to the plot. The default is True.
    legend_loc : str, optional
        String for place where to plot legend, e.g. {'best', 'upper right', 'upper left', 'lower left', 'lower right',
        'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}. The default is 'best'.
    legend_font_size : float, optional
        Percentage of title_font_size to use as font size for legend. The default is 1.0.
    use_log_scale : bool, optional
        Boolean flag to whether or not use logarithm scale for the x axis. The default is False.
    figure_size : list, optional
        List with two int values representing figure size [width, hight]. The default is [15, 10].
    title_font_size : int, optional
        Title font size, which is also used for the axes title sizes (decreased by 4 units). The default is 26.
    save_plot : bool, optional
        Boolean flag to whether or not save the plot as a PNG file. The default is False.
    save_directory : str, optional
        Either "same" for saving in the same folder as the script or "C:\\...\\...\\..." directory where to save figures. The default is "".

    Raises
    ------
    TypeError
        Argument 'data_xy' passed is not a pandas.DataFrame!
    """
    try:
        if ((num_datasets == 1) and not isinstance(data_xy, pd.DataFrame)):
            raise TypeError("TypeError: Argument 'data_xy' passed is not a pandas.DataFrame!")
        fig, ax = plt.subplots(figsize=figure_size)
        ax.grid(True)
        if (use_log_scale is True):
            ax.set_xscale('log')
        if (num_datasets > 1):
            for d in range(num_datasets):
                if (('Avg. ' + root_name[d]) in list(data_xy[d].columns)):
                    data_xy[d].drop(labels=('Avg. ' + root_name[d]), inplace=True)
                if (('Avg. ' + root_name[d] + ' Std') in list(data_xy[d].columns)):
                    data_xy[d].drop(labels=('Avg. ' + root_name[d]), inplace=True)
                end = data_xy[d].shape[1]
                data_xy[d]['Avg. ' + root_name[d]] = data_xy[d].iloc[:, 1:end].mean(axis=1)
                data_xy[d]['Avg. ' + root_name[d] + ' Std'] = data_xy[d].iloc[:, 1:end].std(axis=1)
                xy_data = data_xy[d].copy(deep=True)
                xy_data['Avg. ' + root_name[d] + r' + $\sigma$'] = xy_data['Avg. ' + root_name[d]] + xy_data['Avg. ' + root_name[d] + ' Std']
                xy_data['Avg. ' + root_name[d] + r' - $\sigma$'] = xy_data['Avg. ' + root_name[d]] - xy_data['Avg. ' + root_name[d] + ' Std']
                columns = list(xy_data.columns.values)
                plt.plot(columns[0], columns[-4], data=xy_data, marker=marker_type, linestyle=line_type,
                         markersize=mark_size, color=colors[d], linewidth=line_size)

                if (plot_std_dev):
                    plt.fill_between(columns[0], columns[-2], columns[-1], data=xy_data, label=(r'Avg. ' + root_name[d] + r' $\pm$ $\sigma$'),
                                     linestyle=line_type, color=colors[d], linewidth=float(line_size / (1 * line_size)), alpha=alpha)
        else:
            if (('Avg. ' + root_name) in list(data_xy.columns)):
                data_xy.drop(labels=('Avg. ' + root_name), inplace=True)
            if (('Avg. ' + root_name + ' Std') in list(data_xy.columns)):
                data_xy.drop(labels=('Avg. ' + root_name), inplace=True)
            end = data_xy.shape[1]
            data_xy['Avg. ' + root_name] = data_xy.iloc[:, 1:end].mean(axis=1)
            data_xy['Avg. ' + root_name + ' Std'] = data_xy.iloc[:, 1:end].std(axis=1)
            xy_data = data_xy.copy(deep=True)
            xy_data['Avg. ' + root_name + r' + $\sigma$'] = xy_data['Avg. ' + root_name] + xy_data['Avg. ' + root_name + ' Std']
            xy_data['Avg. ' + root_name + r' - $\sigma$'] = xy_data['Avg. ' + root_name] - xy_data['Avg. ' + root_name + ' Std']
            columns = list(xy_data.columns.values)
            plt.plot(columns[0], columns[-4], data=xy_data, marker=marker_type, linestyle=line_type,
                     markersize=mark_size, color=colors, linewidth=line_size)

            if (plot_std_dev):
                plt.fill_between(columns[0], columns[-2], columns[-1], data=xy_data, label=(r'Avg. ' + root_name + r' $\pm$ $\sigma$'),
                                 linestyle=line_type, color=colors, linewidth=float(line_size / (1 * line_size)), alpha=alpha)

        if (graph_title != ""):
            plt.suptitle(graph_title, fontsize=title_font_size)
        if (x_label == "X Axis"):
            x_label = columns[0]
        if (use_log_scale):
            x_label = x_label + " (log scale)"
        plt.xlabel(x_label, fontsize=(title_font_size - 4))
        plt.ylabel(y_label, fontsize=(title_font_size - 4))
        if (use_x_limits):
            plt.xlim(x_limits[0], x_limits[1])
            if(use_x_ticks):
                ax.xaxis.set_ticks(np.arange(x_limits[0], x_limits[1], x_ticks))
        elif (use_x_ticks):
            ax.xaxis.set_ticks(np.arange(xy_data[columns[0]].min(), xy_data[columns[0]].max(), x_ticks))
        if (use_y_limits):
            plt.ylim(y_limits[0], y_limits[1])
            if (use_y_ticks):
                ax.yaxis.set_ticks(np.arange(y_limits[0], y_limits[1], y_ticks))
        elif (use_y_ticks):
            ax.yaxis.set_ticks(np.arange(xy_data.iloc[:, 1:].min().min(), xy_data.iloc[:, 1:].max().max(), y_ticks))
        if (plot_legend):
            plt.legend(loc=legend_loc, fancybox=True, framealpha=1, shadow=True, borderpad=1,
                       fontsize=int(title_font_size * legend_font_size))
        plt.show()
        if (save_plot):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            file_name = graph_title.replace('\\', "").replace('$', "").replace(' ', "").replace('vs.', "").replace(':', "_")
            if (save_directory == "same"):
                plt.savefig(timestr + file_name + ".png")
            else:
                plt.savefig(save_directory + "/" + timestr + file_name + ".png")
    except Exception as e:
        if (type(e) == TypeError or type(e) == ValueError):
            print(e)
        else:
            print("UnknownError: Problem while running 'plotter_mean()'. Please, review arguments passed...")
            print("ERROR MESSAGE: %s" % (e))


def plotter_grid(vals, fig_size=(8, 8), color='binary', color_lines='black', color_text='black', font_size=16, save_plot=False, fig_name="MyGrid"):
    """
    Function takes as input a grid and plot its values as a matrix.

    Parameters
    ----------
    vals : numpy.ndarray
        Array with values.
    fig_size : tuple, optional
        Size for plot. The default is (8, 8).
    color : str, optional
        Color description. The color is used as color map and MUST be one of the acceptable colors
        from <https://matplotlib.org/stable/tutorials/colors/colormaps.html >. The default is "binary".
    color_lines : str, optional
        Color description from <https://matplotlib.org/stable/tutorials/colors/colormaps.html > for grid lines. The default is "grey".
    color_text : str, optional
        Color description from <https://matplotlib.org/stable/tutorials/colors/colormaps.html > for texts. The default is "black".
    font_size : int, optional
        Size of font. The default is 16.
    save_plot : bool, optional
        Flag to whether or not save the plot. The default is False.
    fig_name : str, optional
        Name for saving figure. The default is "MyGrid".

    Returns
    -------
    None.
    """
    try:
        grid = vals.T
        state = np.arange(0, vals.size).reshape(vals.shape).T  # State indices
        matrix = np.ones((grid.shape[0], grid.shape[1]), dtype=float)
        if (color != 'binary'):
            matrix += grid
        fig, ax = plt.subplots(figsize=fig_size)
        ax.matshow(matrix, cmap=color)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text = r'$V(' + str(state[i][j]) + r')=' + str(grid[i][j]) + r'$'
                ax.text(i, j, text, color=color_text, va='center', ha='center', fontsize=font_size)
        xs = (np.array(list(range(0, grid.shape[0] + 1)), dtype=float) - 0.5)
        ys = (np.array(list(range(0, grid.shape[1] + 1)), dtype=float) - 0.5)
        for y in ys:
            ax.plot(xs, (np.ones(xs.size) * y), marker=None, color=color_lines, linewidth=(font_size / 10))
        for x in xs:
            ax.plot((np.ones(ys.size) * x), ys, marker=None, color=color_lines, linewidth=(font_size / 10))
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        plt.show()
        if (save_plot):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = ax.get_figure()
            fig.savefig(timestr + fig_name + ".png")
    except Exception as e:
        print(e)


def argmax_rand(arr, use_random_argmax=False, rng=np.random.default_rng(59)):
    """
    Method to overcome numpy.argmax() limitation of deterministically return index of first occurrence of maximum value,
    i.e. the method ```argmax_rand()``` identifies the maximum value in an array and all its occurrences and randomly
    selects one amongst those.

    Parameters
    ----------
    arr : numpy.ndarray
        Array with values from which to identify the position of maximum value.
    use_random_argmax : bool, optional
        Flag to choose whether (True) or not (False) to use the random argmax selection amongst multiple equal maximum
        values. The default is False.
    rng : numpy.random._generator.Generator, optional
        Random number generator to choose randomly amongst multiple equal maximum values if option to use random argmax
        is set to True. The default is None.

    Returns
    -------
    indx : int
        The select index position of the identified maximum value.
    """
    try:
        if (use_random_argmax):
            max_vals = np.max(arr)
            max_indx = np.where(arr == max_vals)[0]
            indx = rng.choice(max_indx)
            return indx
        else:
            indx = np.argmax(arr)
            return indx
    except Exception as e:
        print(e)


def action_choice(policy_state, rng=np.random.default_rng(59), use_random_argmax=False, epsilon=None, env=None):
    """
    Function to retrieve action $a$ from the policy $\\pi$ mapping the probability distribution for actions at state $s$.

    Parameters
    ----------
    policy_state : numpy.ndarray
        Array of size # actions with the action probability $\\pi(a|s)$ for state $s$.
    rng : numpy.random._generator.Generator, optional
        Random number generator for action selection. The default is np.random.default_rng(59).
    use_random_argmax : bool, optional
        Flag to whether or not choose argmax action randomly if more than one maximum value action. The default is False.
    epsilon : float, optional
        The exploration probability. Used for selecting actions at random with probability $\\epsilon$. If set to None
        the method chooses the action based on the policy $\\pi(a|s)$. The default is None.
    env : frozen_lake.FrozenLakeEnv, optional
        Object of type FrozenLakeEnv for the environment. The default is None.

    Returns
    -------
    action_selectd : int
        Integer from 0 to (#action-1) representing action $a$ selected from the policy $\\pi(a|s)$ for state $s$.
    """
    try:
        number_actions = len(policy_state)
        if (epsilon is None):
            arr_of_actions = np.arange(number_actions, dtype=int)
            action_selectd = rng.choice(arr_of_actions, p=policy_state)
        else:
            if (rng.random() < epsilon):
                action_selectd = rng.integers(0, env.action_space.n, dtype=int)
            else:
                action_selectd = argmax_rand(policy_state, use_random_argmax=use_random_argmax, rng=rng)
        return action_selectd
    except Exception as e:
        print(e)


def e_soft_policy_state(q_s_a, epsilon=0.01, distributed=False, use_random_argmax=False, rng=np.random.default_rng(59)):
    """
    Helper method to calculate the e-soft/e-greedy policy at state $s$ given the action values $Q(s,a)$.

    Parameters
    ----------
    q_s_a : numpy.ndarray
        Array of size number_acitons with the action values for state $s$.
    epsilon : float
        Epsilon probability of exploration.
    distributed : bool, optional
        Flag to whether or not consider more than one greedy action (Q(s,a_{i}) = Q(s,a_{j})). The default is False.
    use_random_argmax : bool, optional
        Flag to whether or not choose argmax action randomly if more than one maximum value action. The default is False.
    rng : numpy.random._generator.Generator, optional
        Random number generator for action selection. The default is np.random.default_rng(59).

    Returns
    -------
    e_policy_q : numpy.ndarray
        Array of size number_actinos with e-soft policy for state $s$.
    """
    try:
        number_actions = len(q_s_a)
        e_policy_q = np.zeros(number_actions, dtype=float)
        non_greed_p = (epsilon / float(number_actions))
        if (distributed):
            q_max = np.max(q_s_a)
            greedy_acts = np.where(q_s_a == q_max)[0]
        else:
            greedy_acts = np.array([argmax_rand(q_s_a, use_random_argmax=use_random_argmax, rng=rng)])
        greedy_prob = ((1.0 - epsilon) / float(greedy_acts.size))
        for a in range(number_actions):
            if (a in greedy_acts):
                e_policy_q[a] = (greedy_prob + non_greed_p)
            else:
                e_policy_q[a] = non_greed_p
        return e_policy_q
    except Exception as e:
        print(e)


def greedify_policy(policy, use_random_argmax=False, rng=np.random.default_rng(59)):
    """
    Helper method to calculate greedy policy from e-soft/e-greedy policy.

    Parameters
    ----------
    policy : numpy.ndarray
        e-Soft policy to be greedified.
    use_random_argmax : bool, optional
        Flag to whether or not choose argmax action randomly if more than one maximum value action. The default is False.
    rng : numpy.random._generator.Generator, optional
        Random number generator for action selection. The default is np.random.default_rng(59).

    Returns
    -------
    greedy_policy : numpy.ndarray
        Greedy policy.
    """
    try:
        greedy_policy = np.zeros(policy.shape, dtype=float)
        for s in range(len(policy)):
            greedy_a = argmax_rand(policy[s], use_random_argmax=use_random_argmax, rng=rng)
            greedy_policy[s][greedy_a] = 1.0
        return greedy_policy
    except Exception as e:
        print(e)


def generate_episode(policy, env, render=True, rng=np.random.default_rng(59), get_isdone=False):
    """
    Function to generate episode given a policy $\\pi$ and an environment.

    Parameters
    ----------
    policy : numpy.ndarray
        Array of shape (#states, #actions) with the policy actions probabilities $\\pi(a|s)$ for each
        state $s$ and action $a$.
    env : frozen_lake.FrozenLakeEnv
        Object of type FrozenLakeEnv for the environment.
    render : bool, optional
        Flag to whether or not to render the actions $a$ taken in the environment under policy $\\pi$. The default is True.
    rng : numpy.random._generator.Generator, optional
        Random number generator for action selection. The default is np.random.default_rng(59).
    get_isdone : bool, optional
        Flag to whether or not return array with boolean values for state is terminal or not. The default is False.

    Returns
    -------
    states : numpy.ndarray
        Array with the indecies of states visited.
    actions : numpy.ndarray
        Array with the actions $a$ taken under policy $\\pi$ for states $s$ visited.
    rewards : numpy.ndarray
        Array with the rewards $r(s,a,s')$.
    """
    try:
        state = env.reset()  # Returns the initial state s=0
        states, action, reward, isdone = ([], [], [], [])  # Creating empty lists
        while True:
            if (render):
                env.render()
            states.append(state)
            action_s = action_choice(policy_state=policy[state], rng=rng)
            action.append(action_s)
            next_s, r, done, extra = env.step(action_s)
            reward.append(float(r))
            isdone.append(bool(done))
            state = next_s
            if(done):
                break
        states = np.array(states, dtype=int)
        action = np.array(action, dtype=int)
        reward = np.array(reward, dtype=float)
        isdone = np.array(isdone, dtype=bool)
        if (get_isdone):
            return (states, action, reward, isdone)
        else:
            return (states, action, reward)
    except Exception as e:
        print(e)
