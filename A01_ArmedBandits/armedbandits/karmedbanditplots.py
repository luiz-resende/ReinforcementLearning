import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def plotter_func(xy_data, x_label="X Axis", y_label="Y Axis", graph_title="My Plot", use_x_limits=False, x_limits=(0, 100),
                 use_y_limits=False, y_limits=(0, 100), use_x_ticks=False, x_ticks=1, use_y_ticks=False, y_ticks=1,
                 use_log_scale=False, figure_size=[15, 10], title_font_size=26, save_plot=False, save_directory="same"):
    """
    Function created to graphically represent the results of f(x) (up to 28 different functions) for a set of values x=(1,...,n).

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
        color_list = ['red', 'green', 'blue', 'yellow', 'magenta', 'black', 'cyan']
        line_types = ['-', '-.', ':', '--']
        marker_types = ['.', 'o', 'v', 'p', 'D', 's', '+']
        if ((len(columns) - 1) > 28):
            raise ValueError("ValueError: Argument 'xy_data' passed has more then 28 f(x) columns...")
        fig, ax = plt.subplots(figsize=figure_size)
        ax.grid(True)
        if (use_log_scale is True):
            ax.set_xscale('log')
        if (len(columns) < 9):
            for i in range(len(columns) - 1):
                plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[0], markersize=None,
                         color=color_list[i], linewidth=2)
        else:
            for i in range(len(columns) - 1):
                if (i < 7):
                    plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[0],
                             markerfacecolor=None, markersize=None, color=color_list[i], linewidth=2)
                elif ((i >= 7) and (i < 14)):
                    plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[1],
                             markerfacecolor=None, markersize=None, color=color_list[i - 7], linewidth=2)
                    if((i + 1) == (len(columns) - 1)):
                        break
                elif ((i >= 14) and (i < 21)):
                    plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[2],
                             markerfacecolor=None, markersize=None, color=color_list[i - 14], linewidth=2)
                    if((i + 1) == (len(columns) - 1)):
                        break
                elif ((i >= 21) and (i < 28)):
                    plt.plot(columns[0], columns[i + 1], data=xy_data, marker='', linestyle=line_types[3],
                             markerfacecolor=None, markersize=None, color=color_list[i - 21], linewidth=2)
                    if((i + 1) == (len(columns) - 1)):
                        break
        plt.suptitle(graph_title, fontsize=title_font_size)
        if(x_label == "X Axis"):
            x_label = columns[0]
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
        plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize='xx-large')
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
            print("UnknownError: Problem while running 'plotter_func()'. Please, review arguments passed...")
            print("ERROR MESSAGE: %s" % (e))


def find_uniques_idx(arr, sort_type="mergesort"):
    """
    Method to find duplicate elements in an array and return a list of tuples (x, y), where x is the duplicated value and
    y is a list of indices where they accurr in the array.

    Parameters
    ----------
    arr : list or numpy.ndarray
        Array of elements to be tested.
    sort_type : str, optional
        The sorting method to be used in numpy.argsort(). The default is "mergesort".

    Returns
    -------
    repeats : dict
        Dictionary containing duplicate value as key and a list with their respective indices of appearence as key's value.

    Raises
    ------
    TypeError
        The array passed is of a type not supported. Expected type list or type numpy.ndarray and got: type(arr)
        Expected an array of dimension 1, got array of dimension arr.ndim
    """
    try:
        if (type(arr) == list):
            arr = np.array(arr)
        elif (type(arr) != np.ndarray):
            raise TypeError("TypeError: The array passed is of a type not supported. Expected type list or type numpy.ndarray and got: ", type(arr))
        if (arr.ndim > 1):
            raise TypeError("TypeError: Expected an array of dimension 1, got array of dimension ", arr.ndim)
        lst = np.array(range(len(arr)))
        arr = np.array([arr, lst]).T  # Adding column with initial indices
        idx_argsort = np.argsort(arr[:, 0], kind=sort_type)  # Sorting values to have repeats grouped together
        sorted_arr = arr[idx_argsort, :]
        # Returning the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_arr[:, 0], return_counts=True, return_index=True)
        # Remove unique values
        vals = vals[count > 1]
        idx_start = idx_start[count > 1]
        count = count[count > 1]
        repeats = dict()
        # Looking through repeats and retrieving indices
        for v, i, c in zip(vals, idx_start, count):
            idxs = [sorted_arr[i, 1], sorted_arr[i + 1, 1]]
            if c > 2:
                for k in range(2, c):
                    idxs.append(sorted_arr[i + k, 1])
            repeats[v] = idxs
        return repeats
    except Exception as e:
        print(e)
