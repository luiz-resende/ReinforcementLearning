import numpy as np
import matplotlib.pyplot as plt
import time


def plot_grid_world(grid, terminal_states=[], plot_markers=False, color="binary", color_lines="grey", color_text="black",
                    save_plot=False, fig_name="MyGrid", fig_size=(25, 25), font_size=100):
    """
    General helper method created to graphically represent the resulting policy for the gridworld created.

    Parameters
    ----------
    grid : numpy.ndarray
        An array of shape (m, n) with m > 0 representing the gridworld.
    plot_markers : bool, optional
        Flag to choose whether plotting the state value (False) or the arrows (True). The default is False.
    color : str, optional
        Color description. If ```plot_markers=False``` the color is used as color map and MUST be one of the acceptable colors
        from <https://matplotlib.org/stable/tutorials/colors/colormaps.html >. If ```plot_markers=True``` and ```color!="binary"```
        the color is used as the color of the arrows and must be a color from <https://matplotlib.org/stable/gallery/color/named_colors.html >.
        The default is "binary".
    color_lines : str, optional
        Color description from <https://matplotlib.org/stable/tutorials/colors/colormaps.html > for grid lines. The default is "grey".
    color_text : str, optional
        Color description from <https://matplotlib.org/stable/tutorials/colors/colormaps.html > for texts. The default is "black".
    save_plot : bool, optional
        Flag to whether or not save the plot. The default is False.
    fig_name : str, optional
        Name for saving figure. The default is "MyGrid".
    fig_size : tuple, optional
        Tuple (int, int) describing the size of the figure. The default is (25, 25).
    font_size : int, optional
        Integer number for the font sizes used. The default is 100.

    Raises
    ------
    TypeError
        If argument ```grid``` passed is an array of 1 dimension (vector).
    """
    try:
        if(not isinstance(grid, np.ndarray)):
            grid = np.array(grid)
        if (grid.ndim < 2):
            raise TypeError("TypeError: expected an array of shape (m, n) with  m,n>0, but got m=1.")
        if (plot_markers):
            fig, axs = plt.subplots(figsize=fig_size, ncols=grid.shape[1])
            for i, ax in enumerate(axs):
                for j in range(grid.shape[0]):
                    ax.plot([-0.5, -0.5], [(float(j) - 0.5), (float(j) + 0.5)], marker=None, color=color_lines, linewidth=(font_size / 10))
                    ax.plot([0.5, 0.5], [(float(j) - 0.5), (float(j) + 0.5)], marker=None, color=color_lines, linewidth=(font_size / 10))
                    ax.plot([-0.5, 0.5], [(float(j) - 0.5), (float(j) - 0.5)], marker=None, color=color_lines, linewidth=(font_size / 10))
                    ax.plot([-0.5, 0.5], [(float(j) + 0.5), (float(j) + 0.5)], marker=None, color=color_lines, linewidth=(font_size / 10))
                    if (grid[j][i].upper() != 'T' and grid[j][i].lower() != 'o'):
                        ax.plot(j, marker=grid[j][i], color=color, markersize=font_size)
                    else:
                        if (grid[j][i].upper() == 'T' or grid[j][i].lower() == 'o'):
                            ax.text(0, j, r'$\bf{T}$', color='darkgreen', va='center', ha='center', fontsize=int(font_size * 1.5))
                        else:
                            ax.text(0, j, r'$\bf{T}$', color='red', va='center', ha='center', fontsize=int(font_size * 1.5))
                ax.set_axis_off()
                ax.margins(0.0)
                ax.invert_yaxis()
            fig.tight_layout(pad=0.0)
        else:
            grid = grid.T
            intersect_matrix = np.zeros((grid.shape[0], grid.shape[1]), dtype=int)
            if (color != "binary"):
                intersect_matrix += grid
            fig, axs = plt.subplots(figsize=fig_size)
            axs.matshow(intersect_matrix, cmap=color)  # fontsize=(font_size - 6)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if ((j, i) in terminal_states):
                        axs.text(i, j, (r'$\bf{' + str(grid[i][j]) + '}$'), color='darkgreen', va='center', ha='center', fontsize=font_size)
                    else:
                        axs.text(i, j, (str(grid[i][j])), color=color_text, va='center', ha='center', fontsize=font_size)
            xs = (np.array(list(range(grid.shape[0] + 1)), dtype=float) - 0.5)
            ys = (np.array(list(range(grid.shape[1] + 1)), dtype=float) - 0.5)
            for y in ys:
                axs.plot(xs, (np.ones(xs.size) * y), marker=None, color=color_lines, linewidth=(font_size / 10))
            for x in xs:
                axs.plot((np.ones(ys.size) * x), ys, marker=None, color=color_lines, linewidth=(font_size / 10))
            axs.set_axis_off()
            fig.tight_layout(pad=0.0)
        plt.show()
        print("\t" + fig_name + " plotted!")
        if (save_plot):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = axs.get_figure()
            fig.savefig(timestr + fig_name + ".png")
    except Exception as e:
        print("ERROR AT 'PLOT-GRID' METHOD:\n")
        print((str(type(e))[8:-2] + ":"), e)


def plotter_func_double_scale(x_data, y1_data, y2_data, x_label="X Axis", y1_label="Y1 Axis", y2_label="Y2 Axis",
                              graph_title="My Plot", color1="blue", color2="green", line_style1='-', line_style2='-',
                              marker1='', marker2='', line1_width=2, line2_width=2, marker1_size=None, marker2_size=None,
                              figure_size=(15, 10), title_font_size=26, save_plot=False):
    """
    General helper method to plot graph with two y axis scales.

    Parameters
    ----------
    x_data : list, numpy.ndarray
        Array with x values.
    y1_data : list, numpy.ndarray
        Array with y1 values.
    y2_data : list, numpy.ndarray
        Array with y2 values.
    x_label : str, optional
        Label for x axis. The default is "X Axis".
    y1_label : str, optional
        Label for y1 axis. The default is "Y1 Axis".
    y2_label : str, optional
        Label for y2 axis. The default is "Y2 Axis".
    graph_title : str, optional
        Plot title. The default is "My Plot".
    color1 : str, optional
        Color for y1 axis. The default is "blue".
    color2 : str, optional
        Color for y2 axis. The default is "green".
    line_style1 : str, optional
        Line style for y1 axis. The default is '-'.
    line_style2 : str, optional
        Line style for y2 axis. The default is '-'.
    marker1 : str, optional
        Marker style for y1 axis. The default is ''.
    marker2 : str, optional
        Marker style for y2 axis. The default is ''.
    line1_width : int, optional
        Line width for y1 axis. The default is 2.
    line2_width : int, optional
        Line width for y2 axis. The default is 2.
    marker1_size : int, optional
        Marker size for y1 axis. The default is None.
    marker2_size : int, optional
        Marker size for y2 axis. The default is None.
    figure_size : tuple, optional
        Tuple for (width, hight) of figure size. The default is [15, 10].
    title_font_size : int, optional
        Font size used. The default is 26.
    save_plot : bool, optional
        Flag to whether or not save the plot. The default is False.
    """
    try:
        fig, ax1 = plt.subplots(figsize=figure_size)
        ax1.grid(True)
        ax1.set_xlabel(x_label, fontsize=(title_font_size - 4))
        ax1.set_ylabel(y1_label, fontsize=(title_font_size - 4))
        ax1.plot(x_data, y1_data, marker=marker1, linestyle=line_style1, markerfacecolor=None, markersize=marker1_size,
                 color=color1, linewidth=line1_width)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=(title_font_size - 8))
        ax2 = ax1.twinx()
        ax2.grid(True)
        ax2.set_ylabel(y2_label, fontsize=(title_font_size - 4))  # we already handled the x-label with ax1
        ax2.plot(x_data, y2_data, marker=marker2, linestyle=line_style2, markerfacecolor=None, markersize=marker2_size,
                 color=color2, linewidth=line2_width)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=(title_font_size - 8))
        plt.suptitle(graph_title, fontsize=title_font_size)
        fig.tight_layout()
        plt.show()
        if (save_plot):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = plt.get_figure()
            file_name = graph_title.replace('\\', "").replace('$', "").replace(' ', "").replace('vs.', "").replace(':', "_")
            fig.savefig(timestr + file_name + ".png")
    except Exception as e:
        print("ERROR AT 'PLOTTER-FUNC-DOUBLE-SCALE' METHOD:\n")
        print((str(type(e))[8:-2] + ":"), e)
