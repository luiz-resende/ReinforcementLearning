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

# np.set_printoptions(precision=8, threshold=1000, edgeitems=3, linewidth=75, suppress=False, nanstr='nan', infstr='inf', sign='-', floatmode='maxprec')


class GridWorldDP():
    """
    Class implementing a GridWorld environment with methods for solving it through Policy Iteration and Value Iteration
    algorithms from Dynamic Programming method. The instantiation arguments define the grid world shape/size, and the class
    is implemented with the actions {'up', 'down', 'right', 'left'}. When instantiating a class object, the list of terminal
    states must be passed. The arguments for probabilities and ```reward_at_state``` accept both float/int numbers or a
    dictionary  of type {state: reward(state)}, where state=(x,y) (i.e. a tuple with the row and column indices for the
    state). All variables are private, but they can be queried using the method ```var_query()```, and some of them can be
    updated with the method ```var_update()```. All methods have their specific docstrings to facilitate usage (i.e. all
    parameters, arguments and returns are properly explained).

    Attributes
    ----------
    grid_size : int or tuple, optional
        The grid size. If passed int, will construct Grid World of shape (grid_size, grid_size). If passed a tuple (m, n),
        the grid world constructed will have shape (m, n), where $m$ need not be equal to $n$. The default is (5, 5)
    actions : list, optional
        List with the actions available. For the moment, the class bears solution only accepting the fours actions
        {'up', 'down', 'right', 'left'}. The default is ['up', 'down', 'right', 'left'].
    terminal_states : list
        List with the tuples (x,y) for the coordinates of each of the terminal states in the grid. The default is [].
    probability_action : float or dict, optional
        The action probability $\\pi(a|s)$ for the initial policy. If float, the value is repeated for all actions in all
        states. If dict, must be of shape {(state): [pi_a_0, pi_a_1, pi_a_2, pi_a_3]}, where the key state=(x,y) and the key
        value is a list with the probability for each action. The default is 0.25.
    probability_state : float or dict, optional
        The state probability $p(s',r|s,a)$. If float, the value is repeated for all actions in all states' actions, where
        the probability <1.0, the probability of ending in another state is given by ((1.0 - p(s',r|s,a)) / (#actions - 1)).
        If dict, must be of shape {(state): [p_s_a_0, p_s_a_1, p_s_a_2, p_s_a_3]}, where the key state=(x,y) and the key
        value is a list with the probability for each state. passing a float is preferred. The default is 1.0.
    gamma : float, optional
        The $\\gamma$ discounting factor. The default is 1.0.
    reward_at_state : float or dict, optional
        The reward given for changing from state $s$ to $s'$. If float, the reward is assumed equal for every state transition.
        If dict, it must be of shape {(state): [r_s'_0, r_s'_1, r_s'_2, r_s'_3]}, where the key state=(x,y) and the key
        value is a list with the rewards for entering another state (must follow the actions ordering). The dict is preferred
        for cases with variable rewards. The default is -1.0.
    reward_terminal_states : float or list, optinal
        Reward for entering a terminal state (or list with respective float values). However, it is preferred to pass a
        dict in the argument ```reward_at_state```. The default is 0.0.
    reward_boundary : float, optinal
        Reward for hitting a boundary. However, it is preferred to pass a dict in the argument ```reward_at_state``` with
        different rewards for states in the border. The default is -1.0.
    give_reward_terminal : bool, optinal
        Flag to whether or not give the float reward value when getting to a terminal states. Prefer passing a dict at
        ```reward_at_state``` and ignore this argument. The default is False.
    give_reward_boundary : bool, optinal
        Flag to whether or not give the float reward value when hitting a boundary states. Prefer passing a dict at
        ```reward_at_state``` and ignore this argument. The default is False.
    use_random_argmax : bool, optional
        Flag to whether or not select the argmax() randomly when a vector has multiple equal maximum values (intended to
        overcome numpy.argmax() limitation of returning the index where the maximum value first occurs). The default is False.
    default_seed : int, optional
        The default seed used for the general random number generator for the class. The default is 59.
    number_test_episodes : int, optional
        Number of episodes for which to test the current solution found at each learning iteration. The default is 0.
    max_steps_test_policy : int, optional
        The maximum number of steps to allow the current policy to be tested. Used to avoid loop not terminating in initial
        non optimal policies. The default is 100.
    seeds_list : list, optional
        List with integers to be used as seeds for the policy test runs. The default is [].
    number_seeds : int, optional
        Number of seeds to be generated in the case the previous argument is an empty list. The default is 100.
    _number_of_actions : int
        Number of actions available.
    _action_symbol : dict
        Mapping of actions to their symbols for visual represetnation of the policy.
    _state_action_mapping : dict
        Dictionary containing the transition probabilities, next states and rewards, where each key (state=(x,y)) holds a
        list of size # actions with each position storing a tuple (pi_prob_a, next_state, p_prob_s_a, reward).
    _list_all_states : list
        List constaining all the states (x,y) in the grid.
    _list_non_terminal_states : list
        List with all the states in the grid except the terminal states.
    _rng_class : numpy.random._generator.Generator
        A numpy.random.default_rng() object instantiated with the default seed to be used throughout the class to generate
        seeds or to choose actions in the method ```argmax_rand()```.
    _rand_gen_itr : numpy.random._generator.Generator
        numpy.random.default_rng() object used in the algorithm and policy tests.
    _rand_gen_eps : numpy.random._generator.Generator
        numpy.random.default_rng() object used in the algorithm and policy tests.
    _state_values : numpy.ndarray
        Array of shape (grid_size,grid_size) if type(grid_size)=int or shape grid_size if type(grid_size)=tuple containing
        the state values calculated thoughout the algorithms.
    _policy_result : numpy.ndarray
        Array of shape (2,grid_size[0],grid_size[1]) for grid_size=(m,n) containig the policy (selected actions) int numbers
        (i.e. {'up'=0, 'down'=1, 'right'=2, 'left'=3}) in the first position and their symbol represention (i.e. {'up'='^',
        'down'='v', 'right'='>', 'left'='<'}) in the second position.
    iterations_converge : int
        The number of learning iterations the algorithm took to converge to a solution.
    iterations_converge_policy_eval : int
        The number of iterations the policy evaluation took to converge. Zero if ```value_iteration()``` is called.
    policy_testing : list
        List of length # learning iterations, where each entry is a list of length 3 containing the learning iteration number,
        the mean reward and the mean number of steps across the number of test runs.

    Raises
    ------
    ValueError
        Argument ```terminal_states``` must not be empty, otherwise this error is raised
        Argument ```reward_terminal_states```, if passed as a list, it must be of same size as ```terminal_states```.

    Private Methods
    ---------------
    __initialize_args():
        Method to initialize the class variables. It is called whenever the methods ```policy_iteration()``` and
        ```value_iteration()``` are called, such as to allow changes in the contructor arguments after the class is
        instantiated, through the mathod ```var_update()```.
    __helper_is_array(arr):
        Helper method that returns True if argument ```arr``` passed is an array/list, False otherwise.
    __create_state_action_mapping(size_grid, actions_lst, prob_action, prob_state):
        Method creates the transition probabilities, next state and reward dictionary where each key (state=(x,y)) holds a
        list of size # actions with each position storing a tuple (pi_prob_a, next_state, p_prob_s_a, reward).
    __create_map_action_symbol(lst_actions):
        Method that creates the mapping of action symbols used to generate visual aid policy.
    __get_next_state(state_current, action):
        Method that returns the next state (x',y') given a current state (x,y) and an action $a$.
    __get_state_action_reward(state_current, state_next):
        Method returning the reward $r$ from taking action $a$ and changing from state $s$ (x,y) to the state $s'$ (x',y').
    __get_rand_init_state(rng):
        Method that returns a random initial state $s$ (x,y) for testing the policy.
    __get_policy_map():
        Method to extract the policy action probabilites from the variable ```_state_action_mapping```.
    __update_state_action_mapping(actions_probs):
        Method that, given the new policy action probabilities $\\pi(a|s)$, updates them in the variable ```_state_action_mapping```.

    Public Methods
    --------------
    var_update(**kwargs):
        Method to update the value of any number of constructor initialization arguments. To get a list of the names of the
        arguments that can be updated one must only call the function without any argument. When updating a parameter's value,
        it must be passed as key=value in this function, e.g. ```GridWorldDP.var_update(grid_size=(4,4))```.
    var_query(*args):
        Method that returns the value of any of the parameters/variables in the class. To get a list of the names of the
        variables that can be retrieved, one must only call the function without any argument. When getting a parameter's
        value, it must be passed as as a string argument in this function, e.g. ```GridWorldDP.var_query('grid_size')```.
    argmax_rand(arr, use_random_argmax, rng):
        Method to get the argmax in an array either using numpy.argmax() or using a function designed to randomly select
        between the positions where multiple repeating maximum values occur.
    softmax_probability(arr, decimals):
        Method to calculate softmax probability to update action probability values, if the user so desires.
    action_values(state_current, number_actions, gamma, s_values, transition_map):
        Method that returns an array with the action values $Q(a,s)$ for each action $a$ in a state $s$.
    state_sweep(state_values, lst_states, number_actions, gamma, transition_map):
        Method to perform a sweep updating all state values in the grid. It calls function ```action_values()```.
    get_policy(policy, get_symbols, set_new_policy, use_random_argmax, rand_ng):
        Method used to either set the new policy to the class variable (```set_new_policy=True```) or simply to retrieve the
        policy found by either of the algorithms (```set_new_policy=False```).
    policy_test(number_episodes, rand_gen, transition_map, policy, gamma, terminal_states, maximum_number_steps)
        Method to perform policy test episodes and return the mean reward and mean number of steps taken to reach the terminal
        state. It iteratively runs for the desired number of test episodes.
    policy_evaluation(state_values, list_states, number_actions, gamma, transition_map, policy, theta, max_iterations):
        Method implements Iterative Policy Evaluation for a given policy and returns the state values when the iteration
        loop stops after state values converge (difference < theta). The algorithm can also be stoped using a maximum number
        of iterations ```max_iterations```.
    policy_improvement(policy_current, state_values, list_states, number_actions, gamma, transition_map, use_random_argmax, rand_ng):
        Method implements Policy Improvement of a current policy $\\pi$ and returns the new policy and if it is equal or not
        to the previous.
    policy_iteration(theta_diff, max_iterations, on_off_seed, update_action_prob, number_decimals):
        Method implements Policy Iteration algorithm and calls the functions ```policy_evaluation()``` and
        ```policy_improvement()``` sequenctially. By setting ```update_action_prob='softmax'```, the action probabilities are
        updated using softmax. Setting ```update_action_prob='greed'```, the action probabilities are updated greedly with
        probability 1.0 for the action $a$ and 0.0 for $\\forall{A_{t}}\\neq{a}$ at state $s$.
    value_iteration(theta_diff, max_iterations, on_off_seed, number_decimals):
        Method implements Value Iteration algorithm and calls function ```state_sweep()``` successive times until values
        converge.

    Returns
    -------
    Object of class GridWorldDP.
    """

    def __init__(self, grid_size=(5, 5), actions=['up', 'down', 'right', 'left'], terminal_states=[(0, 0), (4, 4)],
                 probability_action=0.25, probability_state=1.0, gamma=1.0, reward_at_state=-1.0, reward_terminal_states=0.0,
                 reward_boundary=-1.0, give_reward_terminal=False, give_reward_boundary=False, use_random_argmax=False,
                 default_seed=59, number_test_episodes=0, max_steps_test_policy=100, seeds_list=[], number_seeds=100):
        """
        Class GridWorldDP constructor.

        Parameters
        ----------
        grid_size : int or tuple, optional
            The grid size. If passed int, will construct Grid World of shape (grid_size, grid_size). If passed a tuple (m, n),
            the grid world constructed will have shape (m, n), where $m$ need not be equal to $n$. The default is (5, 5)
        actions : list, optional
            List with the actions available. For the moment, the class bears solution only accepting the fours actions
            {'up', 'down', 'right', 'left'}. The default is ['up', 'down', 'right', 'left'].
        terminal_states : list
            List with the tuples (x,y) for the coordinates of each of the terminal states in the grid. The default is [].
        probability_action : float or dict, optional
            The action probability $\\pi(a|s)$ for the initial policy. If float, the value is repeated for all actions in all
            states. If dict, must be of shape {(state): [pi_a_0, pi_a_1, pi_a_2, pi_a_3]}, where the key state=(x,y) and the key
            value is a list with the probability for each action. The default is 0.25.
        probability_state : float or dict, optional
            The state probability $p(s',r|s,a)$. If float, the value is repeated for all actions in all states' actions, where
            the probability <1.0, the probability of ending in another state is given by ((1.0 - p(s',r|s,a)) / (#actions - 1)).
            If dict, must be of shape {(state): [p_s_a_0, p_s_a_1, p_s_a_2, p_s_a_3]}, where the key state=(x,y) and the key
            value is a list with the probability for each state. passing a float is preferred. The default is 1.0.
        gamma : float, optional
            The $\\gamma$ discounting factor. The default is 1.0.
        reward_at_state : float or dict, optional
            The reward given for changing from state $s$ to $s'$. If float, the reward is assumed equal for every state transition.
            If dict, it must be of shape {(state): [r_s'_0, r_s'_1, r_s'_2, r_s'_3]}, where the key state=(x,y) and the key
            value is a list with the rewards for entering another state (must follow the actions ordering). The dict is preferred
            for cases with variable rewards. The default is -1.0.
        reward_terminal_states : float or list, optinal
            Reward for entering a terminal state (or list with respective float values). However, it is preferred to pass a
            dict in the argument ```reward_at_state```. The default is 0.0.
        reward_boundary : float, optinal
            Reward for hitting a boundary. However, it is preferred to pass a dict in the argument ```reward_at_state``` with
            different rewards for states in the border. The default is -1.0.
        give_reward_terminal : bool, optinal
            Flag to whether or not give the float reward value when getting to a terminal states. Prefer passing a dict at
            ```reward_at_state``` and ignore this argument. The default is False.
        give_reward_boundary : bool, optinal
            Flag to whether or not give the float reward value when hitting a boundary states. Prefer passing a dict at
            ```reward_at_state``` and ignore this argument. The default is False.
        use_random_argmax : bool, optional
            Flag to whether or not select the argmax() randomly when a vector has multiple equal maximum values (intended to
            overcome numpy.argmax() limitation of returning the index where the maximum value first occurs). The default is False.
        default_seed : int, optional
            The default seed used for the general random number generator for the class. The default is 59.
        number_test_episodes : int, optional
            Number of episodes for which to test the current solution found at each learning iteration. The default is 0.
        max_steps_test_policy : int, optional
            The maximum number of steps to allow the current policy to be tested. Used to avoid loop not terminating in initial
            non optimal policies. The default is 100.
        seeds_list : list, optional
            List with integers to be used as seeds for the policy test runs. The default is [].
        number_seeds : int, optional
            Number of seeds to be generated in the case the previous argument is an empty list. The default is 100.
        """
        # CONSTRUCTOR ARGUMENTS
        self._allowed_argument_update = ['grid_size', 'actions', 'terminal_states', 'reward_at_state', 'reward_boundary',
                                         'reward_terminal_states', 'give_reward_terminal', 'give_reward_boundary',
                                         'probability_action', 'probability_state', 'gamma', 'use_random_argmax',
                                         'default_seed', 'max_steps_test_policy', 'number_test_episodes', 'seeds_list',
                                         'number_seeds']
        self._allowed_argument_query = self._allowed_argument_update + ['number_of_actions', 'action_symbol', 'state_action_mapping',
                                                                        'list_all_states', 'list_non_terminal_states',
                                                                        'rng_class', 'rand_gen_itr', 'rand_gen_eps',
                                                                        'state_values', 'policy_result']
        self._grid_size = grid_size
        self._actions = actions
        if (len(terminal_states) == 0):
            raise ValueError("ValueError: argument 'terminal_states' is empty! Please, pass the terminal states...")
        self._terminal_states = terminal_states
        if (self.__helper_is_array(reward_terminal_states) and (len(reward_terminal_states) < len(self._terminal_states))):
            raise ValueError("ValueError: 'reward_terminal_states' incorrect size. Expected %d and got %d..." % (len(self._terminal_states), len(reward_terminal_states)))
        self._probability_action = probability_action
        self._probability_state = probability_state
        self._gamma = gamma
        self._reward_at_state = reward_at_state
        self._reward_terminal_states = reward_terminal_states
        self._reward_boundary = reward_boundary
        self._give_reward_terminal = give_reward_terminal
        self._give_reward_boundary = give_reward_boundary
        self._use_random_argmax = use_random_argmax
        self._default_seed = default_seed
        self._number_test_episodes = number_test_episodes
        self._max_steps_test = max_steps_test_policy
        self._seeds_list = seeds_list
        self._number_seeds = number_seeds
        # CLASS VARIABLES
        self._number_of_actions = None
        self._action_symbol = dict()
        self._state_action_mapping = dict()
        self._list_all_states = list()
        self._list_non_terminal_states = list()
        self._rng_class = None
        self._rand_gen_itr = None
        self._rand_gen_eps = None
        self._state_values = list()
        self._policy_result = list()
        self.iterations_converge = 0
        self.iterations_converge_policy_eval = list()
        self.policy_testing = list()

    def __initialize_args(self):
        """
        Private method to initialize the class variables when an iteration method is called (either Policy Iteration or Value
        Iteration), allowing the constructing arguments to be updated after class is instantiated.

        Returns
        -------
        True.
            Returns True if variables' initialization is successful.
        """
        try:
            if ((isinstance(self._grid_size, int)) or (isinstance(self._grid_size, float))):
                self._grid_size = (int(self._grid_size), int(self._grid_size))
            elif ((not isinstance(self._grid_size, tuple)) and (not isinstance(self._grid_size, list)) and (not isinstance(self._grid_size, np.ndarray))):
                raise TypeError("TypeError: grid_size should be single int or tuple (int, int). Got %s..." % (type(self._grid_size)))
            elif (((isinstance(self._grid_size, tuple)) or (isinstance(self._grid_size, list)) or (isinstance(self._grid_size, np.ndarray))) and (len(self._grid_size) != 2)):
                raise ValueError("ValueError: expected grid_size of single int or tuple (int, int) and got ", self._grid_size)
            self._number_of_actions = len(self._actions)
            self._action_symbol = self.__create_map_action_symbol(self._actions)
            self._state_action_mapping = self.__create_state_action_mapping(self._grid_size, self._actions,
                                                                            self._probability_action, self._probability_state)
            self._list_all_states = list(self._state_action_mapping.keys())
            self._list_non_terminal_states = [s for s in self._list_all_states if (s not in self._terminal_states)]
            self._rng_class = np.random.default_rng(self._default_seed)
            if (len(self._seeds_list) <= 0):
                self._seeds_list = self._rng_class.integers(1, ((2**63 - 1) - (2**32)), size=self._number_seeds, endpoint=True)
            self._rand_gen_itr = [np.random.default_rng(e) for e in self._seeds_list]
            self._state_values = np.zeros(self._grid_size, dtype=float)  # State values initialized with zero.
            for i, t_state in enumerate(self._terminal_states):
                if (self.__helper_is_array(self._reward_terminal_states)):
                    self._state_values[t_state[0]][t_state[1]] += self._reward_terminal_states[i]
                else:
                    self._state_values[t_state[0]][t_state[1]] += float(self._reward_terminal_states)
            self._policy_result = np.zeros((2, self._grid_size[0], self._grid_size[1]), dtype=int).tolist()
            return True
        except Exception as e:
            print(e)
            return False

    def __helper_is_array(self, arr):
        """
        Private helper method to check if object is instance of list or numpy.ndarray.

        Parameters
        ----------
        arr : any
            Variable to be checked.

        Returns
        -------
        bool
            True if object is instance of either <class 'list'> or <class 'numpy.ndarray'>. False otherwise.
        """
        if (isinstance(arr, list) or isinstance(arr, np.ndarray)):
            return True
        else:
            return False

    def __create_state_action_mapping(self, size_grid, actions_lst, prob_action, prob_state):
        """
        Private method to create dictionary mapping the states to their actions' probabilities, next states and rewards.

        Parameters
        ----------
        size_grid : tuple
            Size of grid in the form (m, n).
        actions_lst : list
            List with available actions $a_i$ for $i=1 to n$.
        prob_action : float or dict
            Probability $\\pi(a|s)$ of selecting an action $a$ under state $s$. It can be a float or a dictionary constructed
            as {(x, y): [pi_a0, pi_a1, ..., pi_an]}.
        prob_state : float or dict
            Probability $p(s',r|s,a)$ of transitioning from state $s$ to $s'$ and getting reward $r$ given action $a$ is
            selected. It can be a float or a dictionary constructed as {(x, y): [p_sa0, p_sa1, ..., p_san]}.

        Returns
        -------
        mapping : dict
            Dictionary with mapping of states, probabilities, next states and rewards.
        """
        try:
            mapping = dict()
            is_dict_p_action = isinstance(prob_action, dict)
            is_dict_p_state = isinstance(prob_state, dict)
            is_float_p_action = (isinstance(prob_action, float) or isinstance(prob_action, int))
            is_float_p_state = (isinstance(prob_state, float) or isinstance(prob_state, int))
            for i in range(size_grid[0]):
                for j in range(size_grid[1]):
                    state = (i, j)
                    mapping[state] = []
                    for a, act in enumerate(actions_lst):
                        next_state = self.__get_next_state(state, act)
                        if (is_float_p_action and is_float_p_state):
                            if (float(prob_action) == 1.0):
                                prob_action = float(prob_action) / self._number_of_actions
                            elif (((prob_action * self._number_of_actions) > 1.0) or ((prob_action * self._number_of_actions) <= 0.0)):
                                raise ValueError("ValueError: sum of action probabilities do not add to 1.0!")
                            mapping[state].append((float(prob_action), next_state, float(prob_state),
                                                   self.__get_state_action_reward(state, next_state)))
                        elif (is_dict_p_action and is_dict_p_state):
                            mapping[state].append((prob_action[state][a], next_state, prob_state[state][a],
                                                   self.__get_state_action_reward(state, next_state)))
                        elif (is_dict_p_action and is_float_p_state):
                            mapping[state].append((prob_action[state][a], next_state, float(prob_state),
                                                   self.__get_state_action_reward(state, next_state)))
                        elif (is_float_p_action and is_dict_p_state):
                            if (float(prob_action) == 1.0):
                                prob_action = float(prob_action) / self._number_of_actions
                            elif (((prob_action * self._number_of_actions) > 1.0) or ((prob_action * self._number_of_actions) <= 0.0)):
                                raise ValueError("ValueError: sum of action probabilities do not add to 1.0!")
                            mapping[state].append((float(prob_action), next_state, prob_state[state][a],
                                                   self.__get_state_action_reward(state, next_state)))
                        elif ((not is_dict_p_action) and (not is_dict_p_state) and (not is_float_p_action) and (not is_float_p_state)):
                            raise TypeError("TypeError: arguments probability_action and probability_state must be float "
                                            + "numbers or dictionaries. Got: %s and %s" % (type(prob_action), type(prob_state)))
            return mapping
        except Exception as e:
            print(e)

    def __create_map_action_symbol(self, lst_actions):
        """
        Private method to create dictionary mapping int values to the four expected actions.

        Parameters
        ----------
        lst_actions : list
            Array with the string representations for the four different actions.

        Raises
        ------
        ValueError
            ValueError: action not recognized! Expected 'up', 'down', 'right' or 'left'.

        Returns
        -------
        temp_dict : dict
            Dictionary mapping int values to the four expected actions.
        """
        try:
            temp_dict = dict()
            for i, a in enumerate(lst_actions):
                if ((a.upper() == 'UP') or (a.upper() == 'U')):
                    temp_dict[i] = '^'
                elif ((a.upper() == 'DOWN') or (a.upper() == 'D')):
                    temp_dict[i] = 'v'
                elif ((a.upper() == 'RIGHT') or (a.upper() == 'R')):
                    temp_dict[i] = '>'
                elif ((a.upper() == 'LEFT') or (a.upper() == 'L')):
                    temp_dict[i] = '<'
                else:
                    raise ValueError("ValueError: action not recognized! Expected 'up', 'down', 'right' or 'left'. Received: %s..." % (a))
            return temp_dict
        except Exception as e:
            print(e)

    def __get_next_state(self, state_current, action):
        """
        Private method to get the next state $s'$ given a current state $s$ and an action $a$ in the gridworld.

        Parameters
        ----------
        state_current : tuple
            Tuple of integers with shape (x, y) representing the coordinates of a given state $s$.
        action : str
            Action $s$ to be taken for setting next possible state $s'$. Must be within {'up', 'down', 'right', 'left',
            'u', 'd', 'r', 'l', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'U', 'D', 'R', 'L', '^', 'v', '>', '<'}.

        Raises
        ------
        ValueError
            ValueError raised if argument 'action' passed is not in {'up', 'down', 'right', 'left', 'u', 'd', 'r', 'l', 'UP',
            'DOWN', 'RIGHT', 'LEFT', 'U', 'D', 'R', 'L', '^', 'v', '>', '<'}.

        Returns
        -------
        tuple
            Tuple of integers with shape (x, y) representing the coordinates of a state $s'$ resulting from taken an action $a$.
        """
        try:
            if ((action.upper() == 'UP') or (action == '^') or (action.upper() == 'U')):
                if (state_current in self._terminal_states):
                    return state_current
                elif (state_current[0] > 0):
                    return ((state_current[0] - 1), state_current[1])
                else:
                    return state_current
            elif ((action.upper() == 'DOWN') or (action == 'v') or (action.upper() == 'D')):
                if (state_current in self._terminal_states):
                    return state_current
                elif (state_current[0] < (self._grid_size[0] - 1)):
                    return ((state_current[0] + 1), state_current[1])
                else:
                    return state_current
            elif ((action.upper() == 'RIGHT') or (action == '>') or (action.upper() == 'R')):
                if (state_current in self._terminal_states):
                    return state_current
                elif (state_current[1] < (self._grid_size[1] - 1)):
                    return (state_current[0], (state_current[1] + 1))
                else:
                    return state_current
            elif ((action.upper() == 'LEFT') or (action == '<') or (action.upper() == 'L')):
                if (state_current in self._terminal_states):
                    return state_current
                elif (state_current[1] > 0):
                    return (state_current[0], (state_current[1] - 1))
                else:
                    return state_current
            else:
                raise ValueError("ValueError: 'action' value not recognized! Expected {'up', 'down', 'right', 'left'}. Received: ", action)
        except Exception as e:
            print(e)

    def __get_state_action_reward(self, state_current, state_next):
        """
        Private method to return the apropriate rewards when constructing the states-actions mapping.

        Parameters
        ----------
        state_current : tuple
            Tuple of integers with shape (x, y) representing the coordinates of a given state $s$.
        state_next : tuple
            Tuple of integers with shape (x, y) representing the coordinates of a next state $s'$.

        Returns
        -------
        float
            The reward $r$ gained from taking action $a$ in state $s$ to state $s'$.
        """
        try:
            if (isinstance(self._reward_at_state, int) or isinstance(self._reward_at_state, float)):
                if ((state_current == state_next) and (state_current in self._terminal_states)):
                    reward = 0.0
                    return reward
                elif ((state_current != state_next) and (state_next in self._terminal_states)):
                    reward = float(self._reward_at_state)
                    if (self._give_reward_terminal and self.__helper_is_array(self._reward_terminal_states)):
                        reward += float(self._reward_terminal_states[self._terminal_states.index(state_next)])
                    elif (self._give_reward_terminal and (not self.__helper_is_array(self._reward_terminal_states))):
                        reward += float(self._reward_terminal_states)
                    return reward
                elif ((state_current == state_next) and (state_next not in self._terminal_states)):
                    reward = float(self._reward_at_state)
                    if (self._give_reward_boundary):
                        reward += float(self._reward_boundary)
                    return reward
                else:  # elif ((state_current != state_next) and (state_next not in self._terminal_states)):
                    reward = float(self._reward_at_state)
                    return reward
            else:
                if ((state_current == state_next) and (state_current in self._terminal_states)):
                    reward = 0.0
                    return reward
                else:
                    reward = self._reward_at_state.get(state_next)
                    return reward
        except Exception as e:
            print(e)

    def __get_rand_init_state(self, rng):
        """
        Private method to randomly select an initial state to test the policy found.

        Parameters
        ----------
        rng : numpy.random._generator.Generator
            Random number generator used to select random initial starting state.

        Returns
        -------
        tuple
            Tuple of integers with shape (x, y) representing the coordinates of chosen initial state.
        """
        try:
            possible_initial_states = self._list_non_terminal_states
            s_0 = rng.integers(0, len(possible_initial_states))
            return possible_initial_states[s_0]
        except Exception as e:
            print(e)

    def __get_policy_map(self):
        """
        Private method to get the action probabilities for policy from the _state_action_mapping.

        Returns
        -------
        policy_probs : dict
            Dictionary with states as keys and the list of action probability $\\pi(a|s)$ for each action $a$ as key values.
        """
        try:
            policy_probs = dict()
            for key in self._state_action_mapping.keys():
                policy_probs[key] = list()
                for a in range(self._number_of_actions):
                    policy_probs[key].append(self._state_action_mapping[key][a][0])
            return policy_probs
        except Exception as e:
            print(e)

    def __update_state_action_mapping(self, actions_probs):
        """
        Helper method to update the action probabilities in the state action transition mapping.

        Parameters
        ----------
        actions_probs : numpy.ndarray
            Array of shape (#states, #actions) with the new action probabilities $\\pi(s|a)$.
        """
        for i, probs in enumerate(actions_probs):
            s = self._list_all_states[i]
            for a, prob in enumerate(probs):
                temp = list(self._state_action_mapping[s][a])
                temp[0] = prob
                self._state_action_mapping[s][a] = tuple(temp)

    def var_update(self, **kwargs):
        """
        Method to update the values of constructor arguments. If function is called without arguments, it returns a list of
        variable names that can be updated.

        Parameters
        ----------
        **kwargs : any
            Variable(s) name(s) and updating value(s).
        """
        try:
            lst_keys = list(kwargs)
            n_keys = len(lst_keys)
            if (n_keys == 0):
                print("The available variables for update are:\n")
                print(self._allowed_argument_update)
            elif (n_keys == 1):
                k = lst_keys[0]
                if (k in self._allowed_argument_update):
                    self.__dict__.update((('_' + key), val) for key, val in kwargs.items())
                else:
                    print("The available variables for update are:\n")
                    print(self._allowed_argument_update)
            else:
                self.__dict__.update((('_' + key), val) for key, val in kwargs.items() if key in self._allowed_argument_update)
                if (not all(elem in self._allowed_argument_update for elem in lst_keys)):
                    print("Some updating variables do not exist! The available variables for update are:\n")
                    print(self._allowed_argument_update)
        except Exception as e:
            print(e)

    def var_query(self, *args):
        """
        Method to query class variable(s) value(s). If function is called without arguments, it returns the list of variable
        names that can be queried.

        Parameters
        ----------
        *args : str
            The names of variables to query.

        Returns
        -------
        any
            It will return the value of a variable or a list with the values of the variables queried.
        """
        try:
            lst_keys = list(args)
            n_keys = len(lst_keys)
            if (n_keys == 0):
                print(self._allowed_argument_query)
                return None
            elif (n_keys == 1):
                key = lst_keys[0]
                if (key in self._allowed_argument_query):
                    return self.__dict__.get(('_' + key))
                else:
                    print("The available variables for query are:\n")
                    print(self._allowed_argument_query)
                    return None
            else:
                query = [self.__dict__.get(('_' + key)) for key in lst_keys if key in self._allowed_argument_query]
                if (all(elem in self._allowed_argument_query for elem in lst_keys)):
                    return query
                else:
                    print("Some queried variables do not exist! The available variables for query are:\n")
                    print(self._allowed_argument_query)
                    return query
        except Exception as e:
            print(e)

    def argmax_rand(self, arr, use_random_argmax=False, rng=None):
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
                max_val = np.max(arr)
                max_indx = np.where(arr == max_val)
                max_indx = max_indx[0]
                prob = (np.ones(max_indx.size) / max_indx.size)
                indx = rng.choice(max_indx, p=prob)
                return indx
            else:
                indx = np.argmax(arr)
                return indx
        except Exception as e:
            print(e)

    def softmax_probability(self, arr, decimals=5):
        """
        Helper method to calculate softmax probability to update action probability $\\pi(s|a)$ in Policy Iteration.

        Parameters
        ----------
        arr : numpy.ndarray
            Array with action values $Q(s,a)$ from which to calculate softmax probabilities.
        decimals : int, optional
            Number of decimals to return in the probabilities array (-1 to not use rounding). The default is 5.

        Returns
        -------
        numpy.ndarray
            DESCRIPTION.

        """
        try:
            softmax_prob = (np.exp(arr) / np.sum(np.exp(arr)))
            if (decimals >= 0):
                return np.round(softmax_prob, decimals=decimals)
            else:
                return softmax_prob
        except Exception as e:
            print(e)

    def action_values(self, state_current, number_actions, gamma, s_values, transition_map):
        """
        Method to compute the action values $Q(s,a)$ for each action $a in A_{s}$ for a state $s in S$ andnext state $s'$.

        Parameters
        ----------
        state_current : tuple
            Tuple of integers with shape (x, y) representing the coordinates of a given state $s$.
        number_actions : int
            Total number of available actions.
        gamma : float
            Discounting $\\gamma$ factor.
        s_values : numpy.ndarray
            2D array with the state values for each of the grid states.
        transition_map : dict
            Dictionary mapping the states to their actions probabilities, rewards and next states.

        Returns
        -------
        Q_s_a : numpy.ndarray
            Array of size number of actions in state $s$ ($|A_{s}|$).
        """
        try:
            Q_s_a = np.zeros(number_actions, dtype=float)
            actions = np.arange(number_actions, dtype=int)
            # BELLMAN EQUATION FOR ACTION VALUE FUNCTION
            for a in actions:
                _, s_prime, p_s_a, r_s_a = transition_map[state_current][a]
                Q_s_a[a] = (p_s_a * (r_s_a + (gamma * s_values[s_prime[0]][s_prime[1]])))
                if (p_s_a < 1.0):
                    actions_other = actions[actions != a]
                    for a_not in actions_other:
                        _, s_a_not, p_s_a_not, r_s_a_not = transition_map[state_current][a_not]
                        if (not isinstance(self._probability_state, dict)):
                            p_s_a_not = ((1.0 - p_s_a) / float(number_actions - 1))
                        Q_s_a[a] += (p_s_a_not * (r_s_a_not + (gamma * s_values[s_a_not[0]][s_a_not[1]])))
            return Q_s_a
        except Exception as e:
            print(e)

    def state_sweep(self, state_values, lst_states, number_actions, gamma, transition_map):
        """
        Method to perform a sweep on the set of states.

        Parameters
        ----------
        state_values : numpy.ndarray
            Array of shape (m, n) representing the states of the gridworld.
        lst_states : list
            List containing the tuples (x, y) for each of the states' coordinates.
        number_actions : int
            Number of available actions.
        gamma : float
            Discounting factor $\\gamma$.
        transition_map : dict
            Dictionary with action transition probabilities, rewards and next states.

        Returns
        -------
        new_state_values : numpy.ndarray
            2D array of shape grid_size (m, n) with the abs. difference between old and new state values to check convergence.
        delta_state_value : numpy.ndarray
            2D array of shape grid_size (m, n) with the new calculated state values.
        """
        try:
            delta_state_value = np.zeros(state_values.shape, dtype=float)
            new_state_values = state_values.copy()
            for s in lst_states:
                Q_s_a = self.action_values(s, number_actions, gamma, state_values, transition_map)
                state_value_previous = state_values[s[0]][s[1]]
                new_state_values[s[0]][s[1]] = np.max(Q_s_a)
                delta_state_value[s[0]][s[1]] = np.abs((state_value_previous - new_state_values[s[0]][s[1]]))
            return new_state_values, delta_state_value
        except Exception as e:
            print(e)

    def get_policy(self, policy=None, get_symbols=True, set_new_policy=False, use_random_argmax=False, rand_ng=None):
        """
        Method to interpret and set/retrieve the policy from the state values results.

        Parameters
        ----------
        policy : numpy.ndarray, optional
            Policy found.
        get_symbols : bool, optional
            Flag to whether method return matrix of symbols (True) or int numbers (False) representing policy actions.
            The default is True.
        set_new_policy : bool, optional
            Flag controlling whether or not to update the class variable containing the policy. The default is False.
        use_random_argmax : bool, optional
            Flag to choose whether (True) or not (False) to use the random argmax selection amongst multiple equal maximum
            values. The default is False.
        rand_ng : numpy.random._generator.Generator, optional
            Random number generator to choose randomly amongst multiple equal maximum values if option to use random argmax
            is set to True. The default is None.

        Returns
        -------
        numpy.ndarray
            Array of shape grid_size (m, n) with the state actions represented as {0: '^', 1: 'v', 2: '>', 3: '<', -1: terminal state 'T'}
        """
        try:
            if (set_new_policy):
                for s in self._list_all_states:
                    if (s not in self._terminal_states):
                        if (policy is None):
                            s_vals = np.zeros(self._number_of_actions, dtype=float)
                            for a in range(self._number_of_actions):
                                _, s_prime, _, _ = self._state_action_mapping[s][a]
                                s_vals[a] += self._state_values[s_prime[0]][s_prime[1]]
                            a_star = self.argmax_rand(s_vals, use_random_argmax, rand_ng)
                        elif (policy is not None):
                            a_star = policy[s[0]][s[1]]
                        self._policy_result[0][s[0]][s[1]] = a_star
                        self._policy_result[1][s[0]][s[1]] = self._action_symbol[a_star]
                    else:
                        self._policy_result[0][s[0]][s[1]] = -1
                        self._policy_result[1][s[0]][s[1]] = 'T'
            if (get_symbols):
                return self._policy_result[1]
            else:
                return self._policy_result[0]
        except Exception as e:
            print(e)

    def policy_test(self, number_episodes, rand_gen, transition_map, policy, gamma, terminal_states, maximum_number_steps=100):
        """
        Method to perform different test episodes with the current policy.

        Parameters
        ----------
        number_episodes : int
            Number of testing episodes.
        rand_gen : numpy.random._generator.Generator
            Random number generator.
        transition_map : dict
            Dictionary with action transition probabilities, rewards and next states.
        policy : numpy.ndarray
            Array of shape (grid_size, grid_size) with the state actions represented as
            {0: up, 1: down, 2: right, 3: left, -1: terminal state}.
        gamma : float
            Discounting factor $\\gamma$.
        terminal_states : list
            List containing the tuples (x, y) for terminal state coordinates.
        maximum_number_steps : int, optional
            Maximum number of iteration steps to force policy test ending if current policy state actions do not converge
            to either of the terminal states. The default is 100.

        Returns
        -------
        tuple
            Tuple (mean cumulative reward, mean cumulative steps).
        """
        try:
            episode_cumulative_reward = np.zeros(number_episodes, dtype=float)
            episode_number_steps = np.zeros(number_episodes, dtype=int)
            for n in range(number_episodes):
                state = self.__get_rand_init_state(rand_gen)
                cumulative_reward = 0.0
                number_steps = 0
                while True:
                    _, state, _, reward = transition_map[state][policy[state[0]][state[1]]]
                    cumulative_reward += ((gamma ** number_steps) * reward)
                    number_steps += 1
                    if ((state in terminal_states) or (number_steps == maximum_number_steps)):
                        break
                episode_cumulative_reward[n] += cumulative_reward
                episode_number_steps[n] += number_steps
            return (np.average(episode_cumulative_reward), np.average(episode_number_steps))
        except Exception as e:
            print(e)

    def policy_evaluation(self, state_values, list_states, number_actions, gamma, transition_map, policy=None, theta=0.000001, max_iterations=1000):
        """
        Method to perform the Policy Evaluation part of the Policy Iteration algorithm.

        Parameters
        ----------
        state_values : numpy.ndarray
            Array of shape (m, n) representing the states of the gridworld.
        lst_states : list
            List containing the tuples (x, y) for each of the states' coordinates.
        number_actions : int
            Number of available actions.
        gamma : float
            Discounting factor $\\gamma$.
        transition_map : dict
            Dictionary with action transition probabilities, rewards and next states.
        policy : dict, optional
            Dictionary with states as keys and the action probabilities $\\pi(a|s)$ as key values. The default is None.
        theta : float, optional
            Very small difference to compare new and old state values and break evaluation loop. The default is 1e-6.
        max_iterations : int, optional
            Maximum number of looping iterations to break policy evaluation if state values do not converge. The default is 1e3.

        Returns
        -------
        state_values : numpy.ndarray
            2D array of shape grid_size (m, n) with the abs. difference between old and new state values to check convergence.
        """
        try:
            if (policy is None):
                policy = self.__get_policy_map()
            itr = 0
            while (itr < max_iterations):
                itr += 1
                new_state_values = np.zeros(state_values.shape, dtype=float)
                for s in list_states:
                    s_value = 0.0
                    # BELLMAN EQUATION FOR STATE VALUE FUNCTION
                    for a, pi_a_s in enumerate(policy[s]):
                        _, s_prime, p_s_a, reward = self._state_action_mapping[s][a]
                        s_value += (pi_a_s * (p_s_a * (reward + (gamma * state_values[s_prime[0]][s_prime[1]]))))
                    new_state_values[s[0]][s[1]] = s_value
                delta_eval = np.maximum(np.zeros(state_values.shape, dtype=float), np.abs(state_values - new_state_values))
                if (np.max(delta_eval) < theta):
                    state_values = new_state_values
                    break
                state_values = new_state_values
            self.iterations_converge_policy_eval.append(itr)
            return state_values
        except Exception as e:
            print(e)

    def policy_improvement(self, policy_current, state_values, list_states, number_actions, gamma, transition_map, use_random_argmax, rand_ng):
        """
        Method to perform Policy Improvement from the Policy Iteration algorithm.

        Parameters
        ----------
        policy_current : numpy.ndarray
            Array of shape grid_size (m, n) representing the current policy (actions selected) for the different states.
        state_values : numpy.ndarray
            Array of shape grid_size (m, n) representing the states of the gridworld.
        lst_states : list
            List containing the tuples (x, y) for each of the states' coordinates.
        number_actions : int
            Number of available actions.
        gamma : float
            Discounting factor $\\gamma$.
        transition_map : dict
            Dictionary with action transition probabilities, rewards and next states.
        use_random_argmax : bool, optional
            Flag to choose whether (True) or not (False) to use the random argmax selection amongst multiple equal maximum
            values. The default is False.
        rand_ng : numpy.random._generator.Generator, optional
            Random number generator to choose randomly amongst multiple equal maximum values if option to use random argmax
            is set to True. The default is None.

        Returns
        -------
        policy_stable : bool
            Flag stating if whether (True) or not (False) the previous and new policies are equal (converged).
        policy_new : numpy.ndarray
            Array of shape grid_size (m, n) representing the new policy (actions selected) for the different states.
        pi_prob_new_softmax : numpy.ndarray
            Array of shape (#states, #actions) with the new action probabilites $\\pi(s|a)$. These are calculated through
            softmax probability using the action values $Q(s,a)$.
        pi_prob_new_greedy : numpy.ndarray
            Array of shape (#states, #actions) with the new action probabilites $\\pi(s|a)$. These are calculated greedly
            setting the probability of selected action to 1.0 and other to 0.0.
        """
        try:
            temp = list()
            policy_stable = True
            policy_new = np.zeros(state_values.shape, dtype=int)
            pi_prob_new_softmax = np.zeros((len(list_states), number_actions), dtype=float)
            pi_prob_new_greedy = np.zeros((len(list_states), number_actions), dtype=float)
            for i, s in enumerate(list_states):
                action_current = policy_current[s[0]][s[1]]
                Q_s_a = self.action_values(s, number_actions, gamma, state_values, transition_map)
                action_new = self.argmax_rand(Q_s_a, use_random_argmax, rand_ng)
                policy_new[s[0]][s[1]] = action_new
                pi_prob_new_softmax[i] = self.softmax_probability(Q_s_a)
                pi_prob_new_greedy[i][action_new] = 1.0
                if (action_new != action_current):
                    policy_stable = False
                temp.append([tuple(np.round(Q_s_a, 2)), action_new])
            return policy_stable, policy_new, pi_prob_new_softmax, pi_prob_new_greedy
        except Exception as e:
            print(e)

    def policy_iteration(self, theta_diff=0.000001, max_iterations=100, max_steps_eval=1000, on_off_seed=False, update_action_prob='greed', number_decimals=-1):
        """
        Method implementing Policy Iteration algorithm. The method calls other two functions, policy_evaluation() and
        policy_improvement(), responsible for the two main parts of the algorithm.

        Parameters
        ----------
        theta_diff : float, optional
             Small threshold determining accuracy of estimation during convergence. The default is 1e-6.
        max_iterations : int, optional
            Parameter for caping the number of iterations 'while' block can run. The default is 100.
        max_steps_eval : int, optional
            Parameter for caping the number of iterations 'while' block can run inside policy evaluation. The default is 1000.
        on_off_seed : bool, optional
            Flag to whether (True) or not (False) ignore the seeds_list and generate seeds on the go. The default is False.
        update_action_prob : str, optional
            String defining the method in which to update the $\\pi(a|s)$ action probabilities. If 'softmax' is selected,
            the action probabilities are updated through the softmax method of the action values. If 'greed' is selected,
            the action probabilities are updated greedly based on the action selected ($\\pi(a|s)=1.0$ and
            $\\pi(A_{t}|s)=0.0\\forall{A_{t}}\\neq{a}$). The default is 'greedy'.
        number_decimals : int, optional
            The number of decimals used to report state value results, or -1 for not to round. The default is -1.

        Raises
        ------
        ValueError
            In the case the list of seeds is passed and its langth is smaller than the number of iterations.

        Returns
        -------
        numpy.ndarray
            Array of shape grid_size (m, n) containing the resulting state values $V(s)$.
        """
        try:
            if (not self.__initialize_args()):
                print("Error initializing the class variables.")
            itr = 0
            policy_test_scores = []
            policy = self._rng_class.integers(min(self._action_symbol.keys()), self._number_of_actions, size=self._grid_size)

            while True:
                itr += 1
                # IF MAXIMUM NUMBER OF ITERATIONS IS REACHED, STOP POLICY ITERATION ALGORITHM
                if (itr > max_iterations):
                    self.iterations_converge = itr - 1
                    break
                if (not on_off_seed):
                    if (len(self._rand_gen_itr) < max_iterations):
                        raise ValueError("ValueError: more iterations then random number generator seeds...")
                    self._rand_gen_eps = self._rand_gen_itr[itr - 1]
                else:
                    self._rand_gen_eps = np.random.default_rng(self._rng_class.integers(1, ((2**63 - 1) - (2**32)), endpoint=True))
                # POLICY EVALUATION
                self._state_values = self.policy_evaluation(state_values=self._state_values, list_states=self._list_all_states,
                                                            number_actions=self._number_of_actions, gamma=self._gamma,
                                                            transition_map=self._state_action_mapping, policy=None,
                                                            theta=theta_diff, max_iterations=max_steps_eval)
                # POLICY IMPROVEMENT
                policy_stable, policy, softmax_acts, greed_acts = self.policy_improvement(policy_current=policy,
                                                                                          state_values=self._state_values,
                                                                                          list_states=self._list_all_states,
                                                                                          number_actions=self._number_of_actions,
                                                                                          gamma=self._gamma,
                                                                                          transition_map=self._state_action_mapping,
                                                                                          use_random_argmax=self._use_random_argmax,
                                                                                          rand_ng=self._rand_gen_eps)
                # UPDATE POLICY ACTION PROBABILITIES (SOFTMAX OR GREEDLY)
                if (update_action_prob == 'softmax'):
                    self.__update_state_action_mapping(softmax_acts)
                elif (update_action_prob == 'greed'):
                    self.__update_state_action_mapping(greed_acts)
                # TESTING POLICY THROUGH n RUNS
                if (self._number_test_episodes > 0):
                    temp_mean_reward, temp_mean_steps = self.policy_test(number_episodes=self._number_test_episodes,
                                                                         rand_gen=self._rand_gen_eps,
                                                                         transition_map=self._state_action_mapping,
                                                                         policy=policy,
                                                                         gamma=self._gamma,
                                                                         terminal_states=self._terminal_states,
                                                                         maximum_number_steps=self._max_steps_test)
                    policy_test_scores.append([itr, temp_mean_reward, temp_mean_steps])
                    self.policy_testing = policy_test_scores
                # IF policy_stable=True, STOP POLICY ITERATION ALGORITHM
                if (policy_stable):
                    self.iterations_converge = itr
                    break
            _ = self.get_policy(policy=policy, get_symbols=True, set_new_policy=True,
                                use_random_argmax=self._use_random_argmax, rand_ng=self._rng_class)
            if (number_decimals < 0):
                return self._state_values
            else:
                return np.round(self._state_values, decimals=number_decimals)
        except Exception as e:
            print(e)

    def value_iteration(self, theta_diff=0.000001, max_iterations=100, on_off_seed=False, number_decimals=-1):
        """
        Method implementing Tabular Value Iteration algorithm.

        Parameters
        ----------
        theta_diff : float, optional
             Small threshold determining accuracy of estimation during convergence. The default is 1e-6.
        max_iterations : int, optional
            Parameter for caping the number of iterations 'while' block can run. The default is 100.
        number_decimals : int, optional
            The number of decimals used to report state value results, or -1 for not to round. The default is -1.

        Raises
        ------
        ValueError
            In the case the list of seeds is passed and its langth is smaller than the number of iterations.

        Returns
        -------
        numpy.ndarray
            Array of shape grid_size (m, n) containing the resulting state values $V(s)$.
        """
        try:
            if (not self.__initialize_args()):
                print("Error initializing the class variables.")
            itr = 0
            policy_test_scores = []
            self.iterations_converge_policy_eval = 0

            while True:
                itr += 1
                # IF MAXIMUM NUMBER OF ITERATIONS IS REACHED, STOP VALUE ITERATION ALGORITHM
                if (itr > max_iterations):
                    self.iterations_converge = itr - 1
                    break
                if (not on_off_seed):
                    if (len(self._rand_gen_itr) < max_iterations):
                        raise ValueError("ValueError: more iterations then random number generator seeds...")
                    self._rand_gen_eps = self._rand_gen_itr[itr - 1]
                else:
                    self._rand_gen_eps = np.random.default_rng(self._rng_class.integers(1, ((2**63 - 1) - (2**32)), endpoint=True))
                # PERFORMING STATE SWEEP
                self._state_values, delta_sweep = self.state_sweep(state_values=self._state_values,
                                                                   lst_states=self._list_all_states,
                                                                   number_actions=self._number_of_actions,
                                                                   gamma=self._gamma,
                                                                   transition_map=self._state_action_mapping)
                delta_state_values = np.maximum(np.zeros(self._grid_size, dtype=float), delta_sweep)
                # TESTING POLICY THROUGH n RUNS
                if (self._number_test_episodes > 0):
                    temp_policy = self.get_policy(policy=None, get_symbols=False, set_new_policy=True,
                                                  use_random_argmax=self._use_random_argmax, rand_ng=self._rand_gen_eps)
                    temp_mean_reward, temp_mean_steps = self.policy_test(number_episodes=self._number_test_episodes,
                                                                         rand_gen=self._rand_gen_eps,
                                                                         transition_map=self._state_action_mapping,
                                                                         policy=temp_policy,
                                                                         gamma=self._gamma,
                                                                         terminal_states=self._terminal_states,
                                                                         maximum_number_steps=self._max_steps_test)
                    policy_test_scores.append([itr, temp_mean_reward, temp_mean_steps])
                    self.policy_testing = policy_test_scores
                # IF STATE VALUES CONVERGE, STOP VALUE ITERATION ALGORITHM
                if (np.max(delta_state_values) < theta_diff):
                    self.iterations_converge = itr
                    break
            _ = self.get_policy(set_new_policy=True, use_random_argmax=self._use_random_argmax, rand_ng=self._rng_class)
            if (number_decimals < 0):
                return self._state_values
            else:
                return np.round(self._state_values, decimals=number_decimals)
        except Exception as e:
            print(e)


# %%

if __name__ == '__main__':
    seeds = [5421010705459837104, 1078615992114306130, 3481173696919315327, 1717966951030518571, 7619115592808984557,
             1478346056475392020, 53676151687212423, 8971119207840718828, 956128101798231187, 8619405905884413185]

    get_plots = True
    # grid_size = (4, 4)
    # terminal_states = [(0, 0), (3, 3)]
    grid_size = 5
    terminal_states = [(0, 0), (4, 4)]
    gw = GridWorldDP(grid_size=grid_size, terminal_states=terminal_states, reward_at_state=-1.0, probability_state=1.0,
                     gamma=1.0, number_test_episodes=5, seeds_list=seeds, max_steps_test_policy=100)
    gw.var_update(use_random_argmax=False)

    s_time = time.perf_counter()
    state_values = gw.value_iteration(theta_diff=0.000001, max_iterations=10, on_off_seed=False, number_decimals=1)
    e_time = time.perf_counter()

    policy = gw.get_policy()
    policy_score = np.array(gw.policy_testing)

    # RESULTS
    print("Number of iterations to converge through Policy Iteration: %d" % (gw.iterations_converge))
    print("\tAverage number of steps to converge in Policy Evaluation: %.1f" % (np.mean(gw.iterations_converge_policy_eval)))
    print(f"Time to find policy through Value Iteration: {e_time - s_time:0.4f} seconds")
    if (get_plots):
        print("State Values:")
        plot_grid_world(state_values, terminal_states=terminal_states, fig_name="state_values", font_size=100)
        print("Policy:")
        plot_grid_world(policy, plot_markers=True, color='black', fig_name="policy")
        print("Policy scores:")
        if (len(policy_score) > 0):
            plotter_func_double_scale(policy_score[:, 0], policy_score[:, 1], policy_score[:, 2], x_label='Number of learning iterations',
                                      y1_label='Mean Cumulative Reward (5 Test Episodes)', y2_label='Average Number of Steps to Terminal State',
                                      graph_title='Value Iteration Policy Scores')


# %%

if __name__ == '__main__':
    get_plots = True
    grid_size = (5, 5)
    terminal_states = [(0, 0), (4, 4)]
    gw = GridWorldDP(grid_size=grid_size, terminal_states=terminal_states, reward_at_state=-1.0, probability_action=0.25,
                     probability_state=1.0, gamma=1.0, number_test_episodes=5, seeds_list=seeds, max_steps_test_policy=100)
    gw.var_update(use_random_argmax=False)

    s_time = time.perf_counter()
    state_values = gw.policy_iteration(theta_diff=0.000001, max_iterations=10, on_off_seed=False, number_decimals=1)
    e_time = time.perf_counter()

    policy = gw.get_policy()
    policy_score = np.array(gw.policy_testing)

    # Results
    print("Number of iterations to converge through Policy Iteration: %d" % (gw.iterations_converge))
    print("\tAverage number of steps to converge in Policy Evaluation: %.1f" % (np.mean(gw.iterations_converge_policy_eval)))
    print(f"Time to find policy through Value Iteration: {e_time - s_time:0.4f} seconds")
    if (get_plots):
        print("State Values:")
        plot_grid_world(state_values, terminal_states=terminal_states, fig_name="state_values", font_size=100)
        print("Policy:")
        plot_grid_world(policy, plot_markers=True, color='black', fig_name="policy")
        print("Policy scores:")
        if (len(policy_score) > 0):
            plotter_func_double_scale(policy_score[:, 0], policy_score[:, 1], policy_score[:, 2], x_label='Number of learning iterations',
                                      y1_label='Mean Cumulative Reward (5 Test Episodes)', y2_label='Average Number of Steps to Terminal State',
                                      graph_title='Value Iteration Policy Scores')


# %%

if __name__ == '__main__':
    get_plots = True
    grid_size = (4, 4)
    terminal_states = [(0, 0)]
    bad_states = [(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 2)]
    rewards = dict()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if ((i, j) in bad_states):
                rewards[(i, j)] = -10.0
            else:
                rewards[(i, j)] = -1.0
    gw = GridWorldDP(grid_size=grid_size, terminal_states=terminal_states, reward_at_state=rewards, probability_action=0.25,
                     probability_state=1.0, gamma=1.0, number_test_episodes=0)
    gw.var_update(use_random_argmax=False)

    s_time = time.perf_counter()
    state_values = gw.policy_iteration(theta_diff=0.000001, max_iterations=7, on_off_seed=True, number_decimals=1)
    e_time = time.perf_counter()

    policy = gw.get_policy()

    # Results
    print("Number of iterations to converge through Policy Iteration: %d" % (gw.iterations_converge))
    print("\tAverage number of steps to converge in Policy Evaluation: %.1f" % (np.mean(gw.iterations_converge_policy_eval)))
    print(f"Time to find policy through Value Iteration: {e_time - s_time:0.4f} seconds")
    if (get_plots):
        print("State Values:")
        plot_grid_world(state_values, terminal_states=terminal_states, fig_name="state_values", font_size=100)
        print("Policy:")
        plot_grid_world(policy, plot_markers=True, color='black', fig_name="policy")
    policy_score = np.array(gw.policy_testing)
    # print("Policy scores:")
    # print(policy_score)
    if (len(policy_score) > 0):
        plotter_func_double_scale(policy_score[:, 0], policy_score[:, 1], policy_score[:, 2], x_label='Number of learning iterations',
                                  y1_label='Mean Cumulative Reward (5 Test Episodes)', y2_label='Average Number of Steps to Terminal State',
                                  graph_title='Value Iteration Policy Scores')
