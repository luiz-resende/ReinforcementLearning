"""
DQN-Atari-Agent

@author: [Luiz Resende Silva](https://github.com/luiz-resende)
@date: Created on Tue Oct 19, 2021

This script implements a Reinforcement Learning agent that executes the DQN
algorithm, training the agent in the ALE Atari environment.

Resources:
----------
Bellemare et. al. (2013) -> https://jair.org/index.php/jair/article/view/10819/25823
Mnih et al.(2013) -> https://arxiv.org/pdf/1312.5602.pdf
Mnih et al.(2015) -> https://www.nature.com/articles/nature14236.pdf
Young and Tian (2019) -> https://arxiv.org/pdf/1903.03176.pdf

"""
import collections
import numpy as np
import random
from typing import Optional, Sequence, Tuple, Union

import gym
import pathlib
import tqdm
import time
import wandb

import torch
from dqn_memory_buffer import MemoryBuffer
from dqn_models_torch import DQNModel
from dqn_wrappers_env import make_atari_env, make_minAtar_env, wrap_atari_env, LazyFrames, MinAtarEnvRGB

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import base64
from pathlib import Path


def show_video(directory):
    """
    Function to display agent recorded test video.

    Notes
    -----
    If you are running this script on Windows, this function might not work because
    of the ``pyvirtualdisplay`` module. To circunvent this problem, just comment-out
    this method, the lines 69-70 below and line 1057 inside ``evaluate_agent()`` method.

    Parameters
    ----------
    directory : str
        Path for directory containing video files.

    Returns
    -------
    None.
    """
    html = []
    for mp4 in Path(directory).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


display = Display(visible=0, size=(1400, 900))
display.start()


class AgentDQN():
    r"""
    Class ``AgentDQN`` implements an reinforcement learning agent with the DQN algorithm for Atari games.

    Parameters
    ----------
    configuration_environment : ``dict``
        A mapping of the necessary information to construct and preprocess an Atari environment.
        It should contain the following keys and their values:
            game_id : ``str``
                Name of the game environment to be built. The default can be ``'Pong-v4'``.
            is_minatar : ``bool``
                Whether it is an Atari environment from ALE (Bellemare et. al., 2013) or the miniaturized
                version (Young and Tian, 2019) with frames of shape (10, 10, n_channels).Please, note
                that if ```is_minatar=True```, only five game environments are available,
                {Asterix, Breakout, Freeway, Seaquest, SpaceInvaders} (with '-v0' or 'v1'). The default can be ``False``.
            render_mode : ``str``
                Render mode for building trining environment ``render_mode='human'`` slows training process.
                The default can be ``'rgb_array'``.
            max_episode_steps_env : ``Union[int, None]``
                Timelimit wrapper. The default can be ``None``.
            no_op_reset_env : ``bool``
                Whether or not use no-op wrapper. The default can be ``True``.
            no_op_max_env : ``int``
                Maximum number of no-op actions. The default can be 30.
            skip_frames_env : ``bool``
                Whether or not to skip a given number of frames. The default can be ``True``.
            skip_frames_env_n : ``int``
                Number of frames to skip. The default can be 4.
            wrap_env : ``bool``
                Whether or not to wrap environment. The default can be ``True``.
            clip_rewards : ``bool``
                Whether or not to clip de rewards to :math:`[-1, 1]`. The default can be ``True``.
            episodic_life : ``bool``
                Whether or not the environment has episodic life. The default can be ``True``.
            scale_frame : ``bool``
                Whether or not to scale frame pixels in the interval :math:`[0.0, 1.0]`.
                The default can be ``False``.
            stack_frames : ``bool``
                Whether or not to stack a given number of frames. The default can be ``True``.
            warp_frames : ``bool``
                Whether or not to warp the frames from their original size to a new shape. The default can be ``True``.
            warp_frames_greyscale : ``bool``
                Whether or not to make frames to greyscale. The default can be ``True``.
    configuration_dqn_models : ``dict``
        A mapping of the necessary information to build the Deep Q-Networks models. The models available
        follow the ones proposed in Mnih et al. (2013), Mnih et al. (2015) and Young and Tian (2019).
        It should contain the following keys and their values:
            in_channels : ``int``
                Number of input channels (or number of 1-channel stacked frames). For example: 4
            out_channel : ``int``
                Number of input channels from first convolutional layer. For example: 16.
            shape_input : ``Union[int, Tuple]``
                The shape of input frames. For example: (84, 84).
            kernel : ``Union[int, Tuple]``
                The size of kernel in the first convolutional layer. For example: (8, 8).
            stride : ``Union[int, Tuple]``
                The size of stride in the first convolutional layer. For example: (4, 4).
            padding : ``Union[int, Tuple]``
                The size of padding in the first convolutional layer. For example: (0, 0).
            out_features_linear : ``int``
                Number of output features in the first linear layer. For example: 256.
            number_actions : ``int``
                Number of actions in the ``env.action_space.n``. For example: 4.
            agent_architecture : ``int``
                The type of architecture, 1 for two convolutional layers (Mnih et al., 2013) or
                2 for three convolutional layers (Mnih et al., 2015). For example: 1, for values above.
            use_batch_norm : ``bool``
                Whether or not use batch normalization between layers. For example: ``False``.
            scale_batch_input : ``float``
                A float number by which to divide the input batches, scaling them. For example: :math:`1.0`.
            device : ``str``
                The device where to move torch tensors and models. It accepts {'cpu', 'gpu', 'cuda'}.
                The default is 'cpu'.
    configuration_optimization : ``dict``
        A mapping of the necessary information to build the model optimizer and the loss function
        to be used in the optimization. It should contain the following keys and their values:
            optim : ``str``
                The optimizer to be used. The implemented training pipeline gives the user the option
                to choose on of the following optimizers: ``'RMSprop'``, ``'Adam'``, ``'Adadelta'``,
                and ``'SGD'``. Any of which instantiate a criteria using the hyperparameters passed
                when instantiating the pipeline class. For example: ``'RMSprop'``.
            loss_criterion : ``str``
                The criterion to calculate the loss. It accepts ``'huber_loss'``, ``'mse_loss'``, ``'l1_loss'``
                and ``'smooth_l1_loss'``. Please, refer to Pytorch official API for further information on the
                different functions, `available here`__. For example: ``'huber_loss'``.
            gamma_disc : ``float``
                Discounting factor gamma. For example: 0.99.
            learn_rate : ``float``
                Learning rate alpha used by optimizer. For example: 0.00025.
            grad_momentum : ``float``
                Gradient momentum used by optimizer. For example: 0.95.
            grad_momentum_square : ``float``
                Squared gradient momentum used by optimizer. For example: 0.95.
            min_sqr_grad : ``float``
                Constant added to the squared gradient in the denominator of the optimizer for stability.
                For example: 0.01.
            epsilon_max : ``float``
                Initial/maximum exploration probability :math:`\epsilon`. For example: 1.0.
            epsilon_min : ``float``
                Final/minimum exploration probability :math:`\epsilon`. For example: 0.1.
            eps_decay_interval : ``int``
                The number of episodes in which the exploration probability will decrease linearly.
                For example: 1000000.
            exponential_decay : ``bool``
                Whether the exploration probability :math:`\epsilon` decays linearly or exponentially.
                If ``False``, :math:`\epsilon` decays linearly at each time step.
            target_network_update : ``int``
                Number of time steps after which to update the target network. For example: 10000.
    configuration_memory_replay : ``dict``
        A mapping of the necessary information to build the experience replay memory buffer.
        It should contain the following keys and their values:
            memory_capacity : ``int``
                Maximum size of the replay memory from which sample experience. For example: 1000000.
            batch_size : ``int``
                Size of experience batch sampled from replay memory. For example: 32.
            initial_memory : ``int``
                Initial number of actions taken using uniform random policy to build initial
                experience replay memory. For example: 50000.
    seed : ``int``, optional
        Seed used for reproducibility. The default is 895359.
    epoch_correspondence : ``int``, optional
        The number of updating iterations (calls to ``optimize_model()``) an epoch corresponds to.
        The default is 50000.
    use_wandb_logging : ``bool``, optional
        Whether or not to use Weights & Biases logging module. The default is ``True``.
    experiment_run_name : ``str``, optional
        Current experiment run name. The default is ``'run_experiment_dqn'``.
    experiment_project_name : ``str``, optional
        The project to which this run is part of. The default is ``'RL_Project'``.
    experiment_run_notes : ``str``, optional
        Some notes to identify run particularities. The default is ``'Testing for some hyperparameter'``.

    __ https://pytorch.org/docs/stable/nn.html#loss-functions

    Arguments
    ---------
    self.__seed : ``int``
    self.game_id : ``str``
    self.render_mode : ``str``
    self.env : ``gym.envs``
    self.env_monitor : ``gym.envs``
    self.action_space_n : ``int``
    self.video_direc : ``str``
    self.device : ``torch.device``
    self.__memory_size : ``int``
    self.buffer_memory : ``__main__.MemoryBuffer``
    self.batch_size : ``int``
    self.initial_memory : ``int``
    self.gamma_disc : ``float``
    self.learn_rate : ``float``
    self.grad_mom : ``float``
    self.grad_mmt_sqr : ``float``
    self.grad_min_sqr : ``float``
    self.eps_max : ``float``
    self.eps_min : ``float``
    self.eps_dec_interval : ``int``
    self.eps_dec : ``float``
    self.eps_dec_exp : ``bool``
    self.update_target_model : ``int``
    self.dqn_policy : ``torch.nn.Module``
    self.dqn_target : ``torch.nn.Module``
    self.optimizer : ``torch.optim``
    self.loss_criterion : ``torch.nn.functional``
    self.epoch_ep_n : ``int``
    self.episodes_scores : ``list``
    self.episodes_losses : ``list``
    self.frames_counter : ``int``
    self.epochs_counter : ``int``
    self.steps_optimize : ``int``
    self.episode_number : ``int``
    self.wandb_logging_on : ``bool``
    self.logger : ``sdk.wandb_run.Run``

    Methods
    -------
    ``seed()``:
        Method sets the global seed and random number generator.
    ``get_tensor()``:
        Method converts observation parameter to torch.Tensor.
    ``sample_action()``:
        Method selects action using either e-greedy or policy_model output.
    ``get_batch_replay()``:
        Method retrieves the list of samples from buffer memory and groups them in specific batches.
    ``optimize_model()``:
        Method retrieves batch of experience and optimizes Deep Q-Network model.
    ``__train_agent_episodes()``:
        Method to call agent's training following an episodic schedule.
    ``__train_agent_frames()``:
        Method to call agent's training following a timestep (frame count) schedule.
    ``train_agent()``:
        Method to call agent's pipeline training.
    ``evaluate_agent()``:
        Method to evaluate trained agent.
    ``save_dqn_models()``:
        Method calls ``torch.save()`` and saves both models to files in a chosen folder.
    ``save_agent_state()``:
        Method saves the agent's entire current state of training, i.e., all the necessary
        class ``AgentDQN`` variables, the models and the memory buffer.
    ``load_dqn_models()``:
        Method calls ``torch.load()`` and loads both models from files in a chosen folder.
    ``load_agent_state()``:
        Method loads the agent's entire state of training from ``save_agent_state()``, for continuing training.
    """

    def __init__(self,
                 configuration_environment: dict,
                 configuration_dqn_models: dict,
                 configuration_optimization: dict,
                 configuration_memory_replay: dict,
                 seed: Optional[int] = 895359,
                 epoch_correspondence: Optional[int] = 50000,
                 use_wandb_logging: Optional[bool] = True,
                 experiment_run_name: Optional[str] = 'run_experiment_dqn',
                 experiment_project_name: Optional[str] = 'RL_Project',
                 experiment_run_notes: Optional[str] = 'Testing for some hyperparameter'
                 ) -> None:
        # SAVING SEED
        self.__seed = seed
        # CREATING THE ENVIRONMENT
        self.__is_minatar = configuration_environment['is_minatar']
        self.render_mode = configuration_environment['render_mode']
        if (self.__is_minatar):
            self.game_id = (r'MinAtar/' + configuration_environment['game_id'])
            self.env = make_minAtar_env(self.game_id,
                                        render_mode=self.render_mode
                                        )  # Creating MinAtar environment
        else:
            if ('NoFrameskip' in configuration_environment['game_id']):
                self.game_id = configuration_environment['game_id']
            else:  # Ensuring the environment is not skiping frames already
                self.game_id = (configuration_environment['game_id'][:-3]
                                + 'NoFrameskip' + configuration_environment['game_id'][-3:])
            self.env = make_atari_env(self.game_id,
                                      render_mode=self.render_mode,
                                      max_episode_steps=configuration_environment['max_episode_steps'],
                                      no_op_reset=configuration_environment['no_op_reset_env'],
                                      no_op_max=configuration_environment['no_op_max_env'],
                                      skip_frames=configuration_environment['skip_frames_env'],
                                      skip_frames_n=configuration_environment['skip_frames_env_n']
                                      )  # Creating Atari ALE environment
        if (configuration_environment['wrap_env']):
            if (self.__is_minatar):
                self.env = MinAtarEnvRGB(self.env,
                                         frame_width=configuration_dqn_models['shape_input'][0],
                                         frame_height=configuration_dqn_models['shape_input'][1],
                                         grayscale=configuration_environment['warp_frames_greyscale']
                                         )
            else:
                self.env = wrap_atari_env(self.env,
                                          clip_rewards=configuration_environment['clip_rewards'],
                                          episodic_life=configuration_environment['episodic_life'],
                                          scale_frame=configuration_environment['scale_frame'],
                                          stack_frames=configuration_environment['stack_frames'],
                                          stack_frames_n=configuration_dqn_models['in_channels'],
                                          warp_frames=configuration_environment['warp_frames'],
                                          warp_frames_greyscale=configuration_environment['warp_frames_greyscale'],
                                          warp_frames_size=configuration_dqn_models['shape_input']
                                          )
        self.action_space_n = self.env.action_space.n
        self.video_direc = (r'./videos/dqn_video_%s_%s' % (self.game_id[:-3].lower(),
                                                           time.strftime('%Y-%m-%d_%Hh%M',
                                                                         time.localtime())))
        if (self.__is_minatar):
            self.env_monitor = gym.wrappers.Monitor(MinAtarEnvRGB(self.env),
                                                    directory=self.video_direc,
                                                    force=True,
                                                    video_callable=lambda episode: True
                                                    )
        else:
            self.env_monitor = gym.wrappers.Monitor(self.env,
                                                    directory=self.video_direc,
                                                    force=True,
                                                    video_callable=lambda episode: True
                                                    )
        # SETTING SEEDS
        self.env.seed(self.__seed)
        random.seed(self.__seed)
        torch.manual_seed(self.__seed)
        # SETTING TORCH DEVICE
        self.device = torch.device('cpu')
        if ((configuration_dqn_models['device'].lower() in ['cuda', 'gpu']) and torch.cuda.is_available()):
            self.device = torch.device('cuda')
        # SETTING DQN MODELS
        self.dqn_policy = DQNModel(in_channels=configuration_dqn_models['in_channels'],
                                   out_channel=configuration_dqn_models['out_channel'],
                                   shape_input=configuration_dqn_models['shape_input'],
                                   kernel=configuration_dqn_models['kernel'],
                                   stride=configuration_dqn_models['stride'],
                                   padding=configuration_dqn_models['padding'],
                                   out_features_linear=configuration_dqn_models['out_features_linear'],
                                   number_actions=self.action_space_n,
                                   agent_architecture=configuration_dqn_models['agent_architecture'],
                                   use_batch_norm=configuration_dqn_models['use_batch_norm'],
                                   scale_batch_input=configuration_dqn_models['scale_batch_input']
                                   ).to(self.device)

        self.dqn_target = DQNModel(in_channels=configuration_dqn_models['in_channels'],
                                   out_channel=configuration_dqn_models['out_channel'],
                                   shape_input=configuration_dqn_models['shape_input'],
                                   kernel=configuration_dqn_models['kernel'],
                                   stride=configuration_dqn_models['stride'],
                                   padding=configuration_dqn_models['padding'],
                                   out_features_linear=configuration_dqn_models['out_features_linear'],
                                   number_actions=self.action_space_n,
                                   agent_architecture=configuration_dqn_models['agent_architecture'],
                                   use_batch_norm=configuration_dqn_models['use_batch_norm'],
                                   scale_batch_input=configuration_dqn_models['scale_batch_input']
                                   ).to(self.device)
        # SETTING THE MODEL OPTIMIZER
        self.gamma_disc = configuration_optimization['gamma_disc']
        self.learn_rate = configuration_optimization['learn_rate']
        self.grad_mmt = configuration_optimization['grad_momentum']
        self.grad_mmt_sqr = configuration_optimization['grad_momentum_square']
        self.grad_min_sqr = configuration_optimization['min_sqr_grad']
        if (configuration_optimization['optimizer'].lower() == 'adam'):
            self.optimizer = torch.optim.Adam(self.dqn_policy.parameters(),
                                              lr=self.learn_rate
                                              )
        elif (configuration_optimization['optimizer'].lower() == 'sgd'):
            self.optimizer = torch.optim.SGD(self.dqn_policy.parameters(),
                                             lr=self.learn_rate,
                                             momentum=self.grad_mmt
                                             )
        elif (configuration_optimization['optimizer'].lower() == 'adadelta'):
            self.optimizer = torch.optim.Adadelta(self.dqn_policy.parameters(),
                                                  rho=self.grad_mmt_sqr,
                                                  lr=self.learn_rate,
                                                  )
        else:  # Uses RMSprop by default
            self.optimizer = torch.optim.RMSprop(self.dqn_policy.parameters(),
                                                 lr=self.learn_rate,
                                                 momentum=self.grad_mmt,
                                                 alpha=self.grad_mmt_sqr,
                                                 eps=self.grad_min_sqr
                                                 )
        self.loss_criterion = configuration_optimization['loss_criterion']
        self.eps_max = configuration_optimization['epsilon_max']
        self.eps_min = configuration_optimization['epsilon_min']
        self.eps_dec_interval = configuration_optimization['eps_decay_interval']
        self.eps_dec = ((self.eps_max - self.eps_min) / float(self.eps_dec_interval))
        self.eps_dec_exp = configuration_optimization['exponential_decay']
        self.update_target_model = configuration_optimization['target_network_update']
        # SETTING EXPERIENCE REPLAY MEMORY
        self.__memory_size = configuration_memory_replay['memory_capacity']
        self.buffer_memory = MemoryBuffer(capacity=self.__memory_size, seed=self.__seed)
        self.batch_size = configuration_memory_replay['sample_batch_size']
        self.initial_memory = configuration_memory_replay['initial_memory']
        self.transition_experience = collections.namedtuple('transition_experience',
                                                            ('s_t0', 'a_t0', 'r_t1', 's_t1')
                                                            )
        # SETTING CONTAINERS FOR STORING TRAINING METRICS
        self.epoch_ep_n = epoch_correspondence
        self.episodes_scores = []
        self.episodes_losses = []
        self.loss_ep = []
        self.frames_counter = 0
        self.epochs_counter = 0
        self.steps_optimize = 0
        self.episode_number = 0
        self.__logging_info = {'experiment_run_name': experiment_run_name,
                               'experiment_project_name': experiment_project_name,
                               'experiment_run_notes': (experiment_run_notes + r' | ' + self.game_id)}
        self.wandb_logging_on = use_wandb_logging
        if (self.wandb_logging_on):
            self.logger = wandb.init(name=experiment_run_name,
                                     project=experiment_project_name,
                                     notes=(experiment_run_notes + r' | ' + self.game_id),
                                     monitor_gym=self.env_monitor
                                     )
            self.logger.define_metric('Loss (t)',
                                      step_metric='Step'
                                      )
            self.logger.define_metric('Avg. Loss (episodes)',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Avg. Score (per epoch)',
                                      step_metric='Epoch'
                                      )
            self.logger.define_metric('Avg. Score Episodes',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Total Score (per episode)',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Avg. Steps per Episode',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Exploration Probability (epsilon)',
                                      step_metric='Step'
                                      )
            self.logger.watch(models=self.dqn_policy)

    def seed(self,
             seed: Optional[Union[int, None]] = None
             ) -> Union[int, None]:
        r"""
        Method sets the global seed and random number generator.

        Parameters
        ----------
        seed : ``Union[int, None]``, optional
            Seed for random number generator. The default is ``None``.

        Returns
        -------
        int
            Seed value if ``seed=None``.

        Raises
        ------
        TypeError:
            TypeError! seed should be integer, instead got {str(type(seed))}...
        """
        if (seed is None):
            return self.__seed
        else:
            if (isinstance(seed, int) or isinstance(seed, int)):
                self.__seed = int(seed)
                random.seed(self.__seed)
                self.env.seed(self.__seed)
                torch.manual_seed(self.__seed)
            else:
                raise TypeError(f'TypeError! seed should be integer, instead got {str(type(seed))}...')

    def get_frame_count(self
                        ) -> int:
        r"""
        Method retrieves the total number of frames used in the training process.

        Returns
        -------
        ``int``
            The total frame count.
        """
        return self.frames_counter

    def get_tensor(self,
                   arr: Union[Sequence, np.ndarray, LazyFrames],
                   lazy_frames: Optional[bool] = True,
                   dtype: Optional[torch.dtype] = torch.float
                   ) -> torch.Tensor:
        r"""
        Method converts a given information from the trasitions observation to torch.Tensor.

        It can convert any of the parameters and move them to the device set. For example, the state
        frame from ``dqn_wrappers_env.LazyFrames`` is converted to ``torch.Tensor`` if ``lazy_frames=True``.

        Parameters
        ----------
        arr : ``Union[Sequence, np.ndarray, LazyFrames]``
            Any of the information from the transition tuple. If is the state observation (Atari frames),
            it is preprocessed to needed shapes and number of channels since frames are kept as LazyFrames objects to
            improve memory efficiency. This method then quickly converts the frames to torch.Tensor of shapes
            ``torch.Size([1, number_channels, frame_height, frame_width])`` when they are required to be used in the model.
        lazy_frames : ``bool``, optional
            Whether or not the input is a stack of LazyFrames object. The default is ``True``.
        dtype : ``Union[torch.bool, torch.float, torch.long, torch.uint8]``, optional
            The data type of the generated tensor. The default is ``torch.float``.

        Returns
        -------
        ``torch.Tensor``
            A tensor of the shape/size of the input unsqueezed to add another dimension. For example, for
            frames, they are converted to torch.Tensor of ``torch.Size([1, number_channels, frame_height, frame_width])``.
        """
        # Converting LazyFrames to NumPy and transposing matrix to have number channels first
        if (lazy_frames):
            arr_tensor = np.array(arr, dtype=str(dtype)[6:])
            arr_tensor = arr_tensor.transpose((2, 0, 1))
            arr_tensor = torch.from_numpy(arr_tensor).type(dtype).to('cpu')
        else:
            arr_tensor = torch.tensor(arr, dtype=dtype, device='cpu')
        return arr_tensor.unsqueeze(0)

    def sample_action(self,
                      obs: torch.Tensor
                      ) -> int:
        r"""
        Method selects action using either :math:`\epsilon`-greedy or ``dqn_policy`` output.

        Parameters
        ----------
        obs : ``torch.Tensor``
            Atari frames preprocessed to match deepmind implementation converted to ``torch.Tensor``.

        Returns
        -------
        act : ``int``
            Integer representing the selected action from ``env.action_space``.
        """
        eps_prob = random.random()
        if (not self.eps_dec_exp):
            eps_iter = np.max([self.eps_min, (self.eps_max - (float(self.frames_counter - 1) * self.eps_dec))])
        else:
            eps_iter = (self.eps_min
                        + ((self.eps_max - self.eps_min) * np.exp(- 1. * (self.frames_counter / self.eps_dec_interval))))
        if (self.wandb_logging_on):
            self.logger.log({'Exploration Probability (epsilon)': eps_iter,
                             'Step': self.frames_counter}
                            )
        if((eps_prob < eps_iter) or (self.frames_counter < self.initial_memory)):
            act = random.randrange(self.action_space_n)
            if (self.save_tensors_to_memory):
                act = self.get_tensor([act], lazy_frames=False, dtype=torch.uint8).to('cpu')

            return act
        else:
            with torch.no_grad():
                act = self.dqn_policy.forward(obs.to(self.device)).max(1)[1].view(1, 1).to('cpu')
                if (not self.save_tensors_to_memory):
                    act = act.item()

                return act

    def optimize_model(self
                       ) -> None:
        r"""
        Method optimizes Deep Q-Network model.

        The method starts by retrieving the list of samples from experience replay memory and
        grouping them into the necessary specific batches. Next, these batches are feed to the
        models, calculating the expectation of the action value-function :math:`Q_{max}(s_{t+1},a_{t+1})`.
        Then, the loss between this expectation and the action value-function approximation
        :math:`\hat{Q}(s_{t}, a_{t})`from the ``dqn_policy`` model is calculated, and the model optimized.

        Parameters
        ----------
        ``None``

        Returns
        -------
        ``None``

        Notes
        -----
        The policy and target agents' parameters are optimized after each :math:`n^{th}` iteration
        by calling this method.
        """
        # Sampling batch of experience
        batch_transitions = self.buffer_memory.random_samples(self.batch_size, look_start=False)
        batch_samples = self.transition_experience(*zip(*batch_transitions))

        # 1 - Grouping current states and converting to tensors - Data type needs to be either torch.float32 or torch.float
        # 2 - Grouping actions - Data type needs to be either torch.long or torch.int64
        # 3 - Grouping rewards - Data type needs to be either torch.float or torch.float32
        # 4 - Grouping next states and converting to tensors - Data type needs to be either torch.float32 or torch.float
        if (self.save_tensors_to_memory):
            batch_s_t0 = torch.cat([s for s in batch_samples.s_t0]).type(torch.float).to(self.device)
            batch_a_t0 = torch.cat([a for a in batch_samples.a_t0]).type(torch.long).to(self.device)
            batch_r_t1 = torch.cat([r for r in batch_samples.r_t1]).type(torch.float).to(self.device)
            batch_s_t1 = torch.cat([s for s in batch_samples.s_t1 if s is not None]).type(torch.float).to(self.device)
            batch_indx = torch.tensor([idx for idx, s in enumerate(batch_samples.s_t1) if s is not None],
                                      dtype=torch.long,
                                      device='cpu')
        else:
            batch_s_t0 = torch.cat([self.get_tensor(s,
                                                    lazy_frames=False,
                                                    dtype=torch.float) for s in batch_samples.s_t0])
            batch_a_t0 = torch.cat([self.get_tensor([a],
                                                    lazy_frames=False,
                                                    dtype=torch.long) for a in batch_samples.a_t0])
            batch_r_t1 = torch.cat([self.get_tensor(r,
                                                    lazy_frames=False,
                                                    dtype=torch.float) for r in batch_samples.r_t1])
            batch_s_t1 = torch.cat([self.get_tensor(s,
                                                    lazy_frames=True,
                                                    dtype=torch.float) for s in batch_samples.s_t1 if s is not None])
            batch_indx = torch.tensor([idx for idx, s in enumerate(batch_samples.s_t1) if s is not None],
                                      dtype=torch.long,
                                      device='cpu')

        # Computing the Q(s_{t}, a_{t}) action values, actions which taken for each batch state according to dqn_policy
        q_vals_policy = self.dqn_policy.forward(batch_s_t0).gather(1, batch_a_t0)

        # Computing the Q_{max}(s_{t+1}, a_{t+1}) and make terminal states values Q_{max}(s_{T}, A) = 0
        q_vals_st1 = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        q_vals_st1[batch_indx] = self.dqn_target.forward(batch_s_t1).max(1)[0].detach()

        # Calculating expectations for s_{t+1}
        q_vals_expect = (q_vals_st1 * self.gamma_disc) + batch_r_t1

        if (self.loss_criterion.lower() == 'mse_loss'):  # Calculating MSE Loss
            loss = torch.nn.functional.mse_loss(q_vals_policy, q_vals_expect.unsqueeze(1))
        elif (self.loss_criterion.lower() == 'l1_loss'):  # Calculating L1 Loss
            loss = torch.nn.functional.l1_loss(q_vals_policy, q_vals_expect.unsqueeze(1))
        elif (self.loss_criterion.lower() == 'smooth_l1_loss'):  # Calculating Smooth-L1 Loss
            loss = torch.nn.functional.smooth_l1_loss(q_vals_policy, q_vals_expect.unsqueeze(1))
        else:  # Calculating Huber Loss
            loss = torch.nn.functional.huber_loss(q_vals_policy, q_vals_expect.unsqueeze(1))

        # Optimizing model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.loss_ep.append(loss.item())
        if (self.wandb_logging_on):
            self.logger.log({'Loss (t)': self.loss_ep[-1],
                             'Step': self.frames_counter}
                            )

    def __train_agent_episodes(self,
                               max_number_episodes: Optional[int] = 1000000,
                               max_number_timestep: Optional[int] = 10000,
                               update_frequency_model: Optional[int] = 1,
                               max_number_training_frames: Optional[Union[int, None]] = None,
                               render: Optional[bool] = False,
                               render_mode: Optional[str] = 'rgb_array'
                               ) -> None:
        r"""
        Method to call agent's training following an episodic schedule.

        Parameters
        ----------
        max_number_episodes : ``int``, optional
            Number of episodes to generate. The default is 1000000.
        max_number_timestep : ``int``, optional
            Maximum number of time steps within a given episode. The default is 10000.
        max_number_training_frames : ``Union[int, None]``, optional
            The hard stop based on number of training timestep iterations (frames generated).
            The default is ``None``.
        update_frequency_model : ``int``, optional
            After how many time steps to update the model, i.e., call ``self.optimize()`` after each
            :math:`n^{th}` time step. The default is 1.
        render : ``bool``, optional
            Whether or not to render the episode while training. The default is ``False``.
        render_mode : ``str``, optional
            Episode rendering mode if ``render=True``. The default is ``'rgb_array'``.

        Returns
        -------
        ``None``
        """
        ep_progress_bar = tqdm.tqdm(range(self.episode_number, max_number_episodes),
                                    desc=r'[Training AgentDQN] ',
                                    unit='ep'
                                    )

        for ep in ep_progress_bar:
            time.sleep(0.0)
            ep_score = 0.0

            s_t0 = self.env.reset()
            if (self.save_tensors_to_memory):
                s_t0 = self.get_tensor(s_t0, lazy_frames=True, dtype=torch.uint8)

            if (render):
                self.env.render(mode=render_mode)

            for t in range(max_number_timestep):
                self.frames_counter += 1

                if (self.save_tensors_to_memory):
                    a_t0 = self.sample_action(s_t0)
                else:
                    a_t0 = np.uint8(self.sample_action(self.get_tensor(s_t0)))

                next_state, r_t1, done, _ = self.env.step(a_t0)  # Making observation

                ep_score += r_t1  # Saving reward to running score

                if (render):
                    self.env.render(mode=render_mode)

                s_t1 = None
                if (self.save_tensors_to_memory):  # Storing tensors, more computationally-efficient
                    r_t1 = self.get_tensor(r_t1, lazy_frames=False, dtype=torch.float16)
                    if (not done):  # Saving only if non-terminal state
                        s_t1 = self.get_tensor(next_state, lazy_frames=True, dtype=torch.uint8)
                else:  # Storing LazyFrames, more memory-efficient
                    r_t1 = np.float16(r_t1)
                    if (not done):  # Saving only if non-terminal state
                        s_t1 = next_state

                # Adding transition tuple to the experience replay memory
                self.buffer_memory.insert(self.transition_experience(s_t0, a_t0, r_t1, s_t1))
                s_t0 = s_t1  # Assigning the new current state

                if (self.frames_counter > self.initial_memory):  # Optimizing after minimum memory achieved

                    self.steps_optimize += 1  # Increasing update_step counter

                    if ((self.steps_optimize % update_frequency_model) == 0):  # Optim. model after a given number of actions
                        self.optimize_model()

                    if ((self.frames_counter % self.update_target_model) == 0):  # Updating target network
                        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

                if (self.wandb_logging_on and ((self.frames_counter % self.epoch_ep_n) == 0)):
                    self.epochs_counter += 1
                    self.logger.log({'Avg. Score (per epoch)': np.mean(self.episodes_scores[-100:]),
                                     'Epoch': self.epochs_counter}
                                    )
                if (done):
                    self.episode_number += 1
                    break  # Ending current episode

            self.episodes_scores.append(ep_score)
            if (self.wandb_logging_on):
                self.logger.log({'Total Score (per episode)': self.episodes_scores[-1],
                                 'Avg. Score Episodes': np.mean(self.episodes_scores[-100:]),
                                 'Avg. Steps per Episode': (self.frames_counter / self.episode_number),
                                 'Episode': self.episode_number}
                                )
            if (self.frames_counter > self.initial_memory):
                self.episodes_losses.append(np.mean(self.loss_ep))
                self.loss_ep = []
                if (self.wandb_logging_on):
                    self.logger.log({'Avg. Loss (episodes)': np.mean(self.episodes_losses),
                                     'Episode': self.episode_number}
                                    )
            ep_progress_bar.set_postfix(AvgSteps=np.round((self.frames_counter / self.episode_number), decimals=3),
                                        AvgRewardEps=np.round(np.mean(self.episodes_scores[-100:]), decimals=3),
                                        RewardMax=np.max(self.episodes_scores),
                                        TotalSteps=self.frames_counter,
                                        )
            if (max_number_training_frames is not None):
                if (self.frames_counter >= max_number_training_frames):
                    break  # Stop training if a maximum number of frames has been generated

    def __train_agent_frames(self,
                             max_number_training_frames: Optional[int] = 50000000,
                             update_frequency_model: Optional[int] = 1,
                             render: Optional[bool] = False,
                             render_mode: Optional[str] = 'rgb_array'
                             ) -> None:
        r"""
        Method to call agent's training following a timestep (frame count) schedule.

        Parameters
        ----------
        max_number_training_frames : ``int``, optional
            The total number of time steps (frames) to be generated for the training. The default is 50000000.
        update_frequency_model : ``int``, optional
            After how many time steps to update the model, i.e., call ``self.optimize()`` after each
            :math:`n^{th}` time step. The default is 1.
        render : ``bool``, optional
            Whether or not to render the episode while training. The default is ``False``.
        render_mode : ``str``, optional
            Episode rendering mode if ``render=True``. The default is ``'rgb_array'``.

        Returns
        -------
        ``None``
        """
        ep_score = 0.0

        frames_progress_bar = tqdm.tqdm(range(max_number_training_frames),
                                        desc=r'[Training AgentDQN] ',
                                        unit=' frames')
        s_t0 = self.env.reset()
        if (self.save_tensors_to_memory):
            s_t0 = self.get_tensor(s_t0, lazy_frames=True, dtype=torch.uint8)

        if (render):
            self.env.render(mode=render_mode)

        for f in frames_progress_bar:
            time.sleep(0.0)

            self.frames_counter += 1

            # Sampling action
            if (self.save_tensors_to_memory):
                a_t0 = self.sample_action(s_t0)
            else:
                a_t0 = np.uint8(self.sample_action(self.get_tensor(s_t0)))

            next_state, r_t1, done, _ = self.env.step(a_t0)  # Making observation

            ep_score += r_t1  # Saving reward to running score

            if (render):
                self.env.render(mode=render_mode)

            s_t1 = None
            if (self.save_tensors_to_memory):  # Storing tensors, more computationally-efficient
                r_t1 = self.get_tensor(r_t1, lazy_frames=False, dtype=torch.float16)
                if (not done):  # Saving only if non-terminal state
                    s_t1 = self.get_tensor(next_state, lazy_frames=True, dtype=torch.uint8)
            else:  # Storing LazyFrames, more memory-efficient
                r_t1 = np.float16(r_t1)
                if (not done):  # Saving only if non-terminal state
                    s_t1 = next_state

            # Adding transition tuple to the experience replay memory
            self.buffer_memory.insert(self.transition_experience(s_t0, a_t0, r_t1, s_t1))
            s_t0 = s_t1  # Assigning the new current state

            if (self.frames_counter > self.initial_memory):  # Optimizing after minimum memory achieved

                self.steps_optimize += 1  # Increasing update_step counter

                if ((self.steps_optimize % update_frequency_model) == 0):  # Optim. model after a given number of actions
                    self.optimize_model()

                if ((self.frames_counter % self.update_target_model) == 0):  # Updating target network
                    self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

            if (self.wandb_logging_on and ((self.frames_counter % self.epoch_ep_n) == 0)):
                self.epochs_counter += 1
                self.logger.log({'Avg. Score (per epoch)': np.mean(self.episodes_scores[-100:]),
                                 'Epoch': self.epochs_counter}
                                )
            if (done):
                self.episode_number += 1
                s_t0 = self.env.reset()
                if (self.save_tensors_to_memory):
                    s_t0 = self.get_tensor(s_t0, lazy_frames=True, dtype=torch.uint8)

                self.episodes_scores.append(ep_score)
                ep_score = 0

                if (self.wandb_logging_on):
                    self.logger.log({'Total Score (per episode)': self.episodes_scores[-1],
                                     'Avg. Score Episodes': np.mean(self.episodes_scores[-100:]),
                                     'Avg. Steps per Episode': (self.frames_counter / self.episode_number),
                                     'Episode': self.episode_number}
                                    )
                if (self.frames_counter > self.initial_memory):
                    self.episodes_losses.append(np.mean(self.loss_ep))
                    self.loss_ep = []
                    if (self.wandb_logging_on):
                        self.logger.log({'Avg. Loss (episodes)': np.mean(self.episodes_losses[-100:]),
                                         'Episode': self.episode_number}
                                        )
                frames_progress_bar.set_postfix(AvgSteps=np.round((self.frames_counter / self.episode_number), decimals=3),
                                                AvgRewardEps=np.round(np.mean(self.episodes_scores[-100:]), decimals=3),
                                                RewardMax=np.max(self.episodes_scores)
                                                )

    def train_agent(self,
                    train_in_episodes: Optional[bool] = False,
                    max_number_training_frames: Optional[Union[int, None]] = 50000000,
                    max_number_episodes: Optional[int] = 1000000,
                    max_number_timestep: Optional[int] = 10000,
                    update_frequency_model: Optional[int] = 1,
                    render: Optional[bool] = False,
                    render_mode: Optional[str] = 'rgb_array',
                    save_tensors_in_memory_buffer: Optional[bool] = False,
                    load_agent_state: Optional[bool] = False,
                    load_agent_info: Optional[Union[None, Tuple]] = None,
                    save_iterruption: Optional[bool] = True
                    ) -> Union[Sequence, None]:
        r"""
        Method to call agent's pipeline training.

        Parameters
        ----------
        train_in_episodes : ``bool``, optional
            Whether to train generating cycles of episodes (``True``) or train continuously generating
            frames (``False``). The default is ``False``.
        max_number_training_frames : ``Union[int, None]``, optional
            The total number of time steps (frames) to be generated for the training. If ``train_in_episodes=True``,
            it will be used as a secondary stopping criteria. Set ``max_number_training_frames=None`` to disable this
            secondary stopping criteria. The default is 50000000.
        max_number_episodes : ``int``, optional
            Maximum number of episodes to be generated if ``train_in_episodes=True``. The default is 1000000.
        max_number_timestep : ``int``, optional
            Maximum number of time steps within an episode if ``train_in_episodes=True``. The default is 10000.
        update_frequency_model : ``int``, optional
            After how many time steps to update the model, i.e., call ``self.optimize()`` after each
            :math:`n^{th}` time step. The default is 1.
        render : ``bool``, optional
            Whether or not to render the episode while training. The default is ``False``.
        render_mode : ``str``, optional
            Episode rendering mode if ``render=True``. The default is ``'rgb_array'``.
        save_tensors_in_memory_buffer : ``bool``, optional
            Whether or not to save state observations in the experience replay memory buffer already converted
            to torch.Tensors. The default is ``False``.
                NOTE: set it to ``True`` if enough RAM is available).
        load_agent_state : ``bool``, optional
            Whether or not the agent's entire state of a previous training is to be load and the training continue
            (``True``), or if it should start the training process without previous information (``False``).
            The default is ``False``.
        load_agent_info : ``Tuple``, optional
            If ``load_agent_state=True``, the information to retrieve the agent's state shoud be passed in a tuple
            containing (base_file_name: str, postfix: str, load_target_model: bool), and the definition of these
            arguments can be found in the description of the method ``load_agent_state``. The default is ``None``.
        save_iterruption : ``bool``, optional
            Whether or not to save the agents current state of training if it is interrupted by some exception.
            The default is ``True``.

        Returns
        -------
        ``Union[list, None]``
            Returns sequence with the scores per training episode or ``None`` if some exception is caught.
        """
        self.save_tensors_to_memory = save_tensors_in_memory_buffer
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

        if (load_agent_state):
            if (load_agent_info is not None):
                self.load_agent_state(load_agent_info[0], load_agent_info[1], load_agent_info[2])
            else:
                raise ValueError('ValueError: Information load_agent_info is required, but got None instead!')

        try:
            if (train_in_episodes):
                self.__train_agent_episodes(max_number_episodes=max_number_episodes,
                                            max_number_timestep=max_number_timestep,
                                            update_frequency_model=update_frequency_model,
                                            max_number_training_frames=max_number_training_frames,
                                            render=render,
                                            render_mode=render_mode
                                            )
            else:
                self.__train_agent_frames(max_number_training_frames=max_number_training_frames,
                                          update_frequency_model=update_frequency_model,
                                          render=render,
                                          render_mode=render_mode
                                          )

            self.env.close()
            if (self.wandb_logging_on):
                self.logger.finish()

            return self.episodes_scores

        except (Exception, KeyboardInterrupt) as e:
            print(e)
            if (save_iterruption):
                print("\nSaving model before exiting training process...")
                self.save_agent_state(base_file_name='agent_training_interrupted',
                                      postfix='',
                                      save_experience_replay=True)

            self.env.close()
            if (self.wandb_logging_on):
                self.logger.finish()

            return None

    def evaluate_agent(self,
                       number_episodes: Optional[int] = 50,
                       render: Optional[bool] = True,
                       render_mode: Optional[str] = 'human'
                       ) -> None:
        r"""
        Method to evaluate trained agent.

        Parameters
        ----------
        number_episodes : ``int``, optional
            Number of episode to test the agent. The default is 1000.
        render : ``bool``, optional
            Flag to whether or not render episode. The default is ``True``.
        render_mode : ``str``, optional
            Episode rendering mode. The default is ``'human'``.

        Returns
        -------
        ``None``
        """
        episodes_scores_eval = []

        for e in range(number_episodes):
            score_episode_eval = 0
            s_t0 = self.env_monitor.reset()
            s_t0 = self.get_tensor(s_t0)
            d_t1 = False

            while (not d_t1):
                a_t0 = self.dqn_policy.forward(s_t0).max(1)[1].item()
                s_t1, r_t1, d_t1, _ = self.env_monitor.step(a_t0)
                s_t0 = self.get_tensor(s_t1)
                score_episode_eval += r_t1

                if (render):
                    self.env_monitor.render(mode=render_mode)
                    time.sleep(0.05)

                if (d_t1):
                    episodes_scores_eval.append(score_episode_eval)
                    print(f"Finished Episode {(e + 1)} with reward {score_episode_eval}")
                    break

        print(f'Final average reward {np.round(np.mean(episodes_scores_eval), decimals=3)} +/- '
              + f'{np.round(np.std(episodes_scores_eval), decimals=3)}')

        self.env_monitor.close()
        show_video(self.video_direc)

    def save_dqn_models(self,
                        path: Optional[str] = r'./saved_models/',
                        postfix: Optional[str] = ''
                        ) -> None:
        r"""
        Method calls ``torch.save()`` and saves both the policy and target trained networks to files in a chosen folder.

        Parameters
        ----------
        path : ``str``, optional
            Path of directory to save models. The default is r'./saved_models/'.
        postfix : ``str``, optional
            A postfix string to be appended to the end of the file name. The default is ''.
        """
        base_name = path
        if ((postfix != '') and (postfix[0] != '_')):
            postfix = '_' + postfix
        if (path == r'./saved_models/'):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            base_name = (path + time.strftime('%Y-%m-%d_%Hh%M', time.localtime()))
        policy_name = (r'%s_policy_q_net%s.pt' % (base_name, postfix))
        target_name = (r'%s_target_q_net%s.pt' % (base_name, postfix))
        torch.save(self.dqn_policy, policy_name)
        torch.save(self.dqn_target, target_name)

    def save_agent_state(self,
                         base_file_name: Optional[str] = 'agent_saved',
                         postfix: Optional[str] = '',
                         save_experience_replay: Optional[bool] = False
                         ) -> None:
        r"""
        Method designed to save the agent's entire current state of training.

        Calling this method will converted all agent's information and saved to a ``'.pkl'``
        pickle file.

        Notes
        -----
        Given the size of the experiance replay memory and the amounto RAM necessary to convert and store
        the saved transition tuples, the method caps its size to at most 100K most recent experienced time
        steps (if its size is greater than 100K).

        Parameters
        ----------
        base_file_name : ``str``, optional
            The base name to be given to the files. It should contain the directory path and some common shared
            name. It will be later added the time stamp and the different file identifiers. The default is ``'saved_agent'``.
        postfix : ``str``, optional
            Some string information the user would like to add to all the names. The default is ``''``.
        save_experience_replay : ``bool``, optional
            Whether or not to save the experience replay memory. The default is ``False``.

        Returns
        -------
        ``None``
        """
        base_file_name = base_file_name + '_' + time.strftime('%Y-%m-%d_%Hh%M', time.localtime())
        if ((postfix != '') and (postfix[0] != '_')):
            postfix = '_' + postfix
        if (self.buffer_memory.size > 100000):
            self.buffer_memory = MemoryBuffer(data=self.buffer_memory.tolist()[-100000:], capacity=1000000)
        lists_transitions = self.transition_experience(*zip(*self.buffer_memory.tolist()))
        experience = []
        if (save_experience_replay):
            if (not self.save_tensors_to_memory):
                experience.append(np.array(lists_transitions.s_t0, dtype='uint8'))
                experience.append(np.array(lists_transitions.a_t0, dtype='uint8'))
                experience.append(np.array(lists_transitions.r_t1, dtype='float16'))
                experience.append([s.tolist(dtype='uint8') if s is not None else None for s in lists_transitions.s_t1])
            else:
                experience.append(list(lists_transitions.s_t0))
                experience.append(list(lists_transitions.a_t0))
                experience.append(list(lists_transitions.r_t1))
                experience.append(list(lists_transitions.s_t1))

        agent_state = {'seed': self.__seed,
                       'is_MinAtar_env': self.__is_minatar,
                       'env_id': self.game_id,
                       'env_render_mode': self.render_mode,
                       'environment_obj': self.env,
                       'video_directory': self.video_direc,
                       'torch_device': str(self.device),
                       'policy_model': self.dqn_policy,
                       'target_model': self.dqn_target,
                       'discount_factor': self.gamma_disc,
                       'learning_rate': self.learn_rate,
                       'gradient_momentum': self.grad_mmt,
                       'square_gradient_momentum': self.grad_mmt_sqr,
                       'minimum_squared_gradient': self.grad_min_sqr,
                       'maximum_exploration': self.eps_max,
                       'minimum_exploration': self.eps_min,
                       'exploration_annealing': self.eps_dec,
                       'use_exponential_decay': self.eps_dec_exp,
                       'policy_optimizer': self.optimizer,
                       'loss_function_used': self.loss_criterion,
                       'target_model_update': self.update_target_model,
                       'replay_memory_size': self.__memory_size,
                       'batch_size': self.batch_size,
                       'start_experience_replay': self.initial_memory,
                       'experience_replay_transitions': experience,
                       'frames_to_epoch_correspondence': self.epoch_ep_n,
                       'list_episode_scores': self.episodes_scores,
                       'list_episode_losses': self.episodes_losses,
                       'number_frames_trained': self.frames_counter,
                       'number_epochs_trained': self.epochs_counter,
                       'number_steps_optimize': self.steps_optimize,
                       'number_episodes_trained': self.episode_number,
                       'use_wandb_logging': self.wandb_logging_on,
                       'logging_information': self.__logging_info,
                       'experience_saved_as_tensors': self.save_tensors_to_memory,
                       }
        torch.save(agent_state, f'{base_file_name}_agent_state{postfix}.pkl')

        del experience, lists_transitions, agent_state, base_file_name

    def load_dqn_models(self,
                        path_name: Optional[str] = r'./saved_models/',
                        postfix: Optional[str] = '',
                        load_target_model: Optional[bool] = False
                        ) -> None:
        r"""
        Method calls ``torch.load()`` and loads both the policy and target trained networks from files in a chosen folder.

        Notes
        -----
        This method should be called only after the models have been created.

        Parameters
        ----------
        path_name : ``str``, optional
            Path of directory to load models. The default is r'./saved_models/'.
        postfix : ``str``, optional
            A postfix string that is appended after ``'net_'`` and before ``'.pt'`` . The default is ``''``.
        load_target : ``bool``, optional
            Wheter or not to load target model. The default is False.

        Notes
        -----
        Method assumes saved models has the model names given by ``saved_dqn_agents()``.
        """
        if ((postfix != '') and (postfix[0] != '_')):
            postfix = '_' + postfix
        policy_name = (r'%s_policy_q_net%s.pt' % (path_name, postfix))
        self.dqn_policy = torch.load(policy_name)
        if (load_target_model):
            target_name = (r'%s_target_q_net%s.pt' % (path_name, postfix))
            self.dqn_target = torch.load(target_name)

    def load_agent_state(self,
                         base_file_name: str,
                         postfix: Optional[str] = '',
                         load_target_model: Optional[bool] = False
                         ) -> None:
        r"""
        Method designed to load the agent's entire state of training, for continuing training.

        Notes
        -----
        The method assumes the naming given in ``save_agent_state()`` is mainteined.

        Parameters
        ----------
        base_file_name : ``str``
            The base name from the files. It should contain the directory path and some common shared
            name (including time stamp). It will be later added the different file identifiers.
        postfix : ``str``, optional
            Some string information the files have in common and is appended after the identifiers
            and before file extensions. The default is ``''``.
        load_target_model : ``bool``, optional
            Wheter or not to load target model. The default is ``False``.

        Returns
        -------
        ``None``
        """
        if ((postfix != '') and (postfix[0] != '_')):
            postfix = '_' + postfix
        agent_state = torch.load(f'{base_file_name}_agent_state{postfix}.pkl')
        # SEED
        self.__seed = int(agent_state['seed'])
        # ENVIRONMENT
        self.__is_minatar = bool(agent_state['is_MinAtar_env'])
        self.game_id = str(agent_state['env_id'])
        self.render_mode = str(agent_state['env_render_mode'])
        self.env = agent_state['environment_obj']
        self.action_space_n = self.env.action_space.n
        self.video_direc = str(agent_state['video_directory'])
        if (self.__is_minatar):
            self.env_monitor = gym.wrappers.Monitor(MinAtarEnvRGB(self.env),
                                                    directory=self.video_direc,
                                                    force=True,
                                                    video_callable=lambda episode: True
                                                    )
        else:
            self.env_monitor = gym.wrappers.Monitor(self.env,
                                                    directory=self.video_direc,
                                                    force=True,
                                                    video_callable=lambda episode: True
                                                    )
        # SETTING SEEDS
        self.env.seed(self.__seed)
        random.seed(self.__seed)
        torch.manual_seed(self.__seed)
        # SETTING TORCH DEVICE
        self.device = torch.device('cpu')
        if ((str(agent_state['torch_device']).lower() in ['cuda', 'gpu']) and torch.cuda.is_available()):
            self.device = torch.device('cuda')
        # SETTING DQN MODELS
        self.dqn_policy = agent_state['policy_model']
        if (load_target_model):
            self.dqn_target = agent_state['target_model']
        # SETTING THE MODEL OPTIMIZER
        self.gamma_disc = float(agent_state['discount_factor'])
        self.learn_rate = float(agent_state['learning_rate'])
        self.grad_mmt = float(agent_state['gradient_momentum'])
        self.grad_mmt_sqr = float(agent_state['square_gradient_momentum'])
        self.grad_min_sqr = float(agent_state['minimum_squared_gradient'])
        self.eps_max = float(agent_state['maximum_exploration'])
        self.eps_min = float(agent_state['minimum_exploration'])
        self.eps_dec = float(agent_state['exploration_annealing'])
        self.eps_dec_exp = bool(agent_state['use_exponential_decay'])
        self.optimizer = agent_state['policy_optimizer']
        self.loss_criterion = str(agent_state['loss_function_used'])
        self.update_target_model = int(agent_state['target_model_update'])
        # SETTING EXPERIENCE REPLAY MEMORY
        self.__memory_size = int(agent_state['replay_memory_size'])
        self.buffer_memory = MemoryBuffer(capacity=self.__memory_size, seed=self.__seed)
        self.batch_size = int(agent_state['batch_size'])
        self.initial_memory = int(agent_state['start_experience_replay'])
        self.transition_experience = collections.namedtuple('transition_experience',
                                                            ('s_t0', 'a_t0', 'r_t1', 's_t1')
                                                            )
        experience = agent_state['experience_replay_transitions']
        # TRAINING METRICS
        self.epoch_ep_n = int(agent_state['frames_to_epoch_correspondence'])
        self.episodes_scores = list(agent_state['list_episode_scores'])
        self.episodes_losses = list(agent_state['list_episode_losses'])
        self.frames_counter = int(agent_state['number_frames_trained'])
        self.epochs_counter = int(agent_state['number_epochs_trained'])
        self.steps_optimize = int(agent_state['number_steps_optimize'])
        self.episode_number = int(agent_state['number_episodes_trained'])
        self.wandb_logging_on = bool(agent_state['use_wandb_logging'])
        self.__logging_info = agent_state['logging_information']
        if (self.wandb_logging_on):
            self.logger = wandb.init(name=self.__logging_info['experiment_run_name'],
                                     project=self.__logging_info['experiment_project_name'],
                                     notes=self.__logging_info['experiment_run_notes'],
                                     monitor_gym=self.env_monitor
                                     )
            self.logger.define_metric('Loss (t)',
                                      step_metric='Step'
                                      )
            self.logger.define_metric('Avg. Loss (episodes)',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Avg. Score (per epoch)',
                                      step_metric='Epoch'
                                      )
            self.logger.define_metric('Avg. Score Episodes',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Total Score (per episode)',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Avg. Steps per Episode',
                                      step_metric='Episode'
                                      )
            self.logger.define_metric('Exploration Probability (epsilon)',
                                      step_metric='Step'
                                      )
            self.logger.watch(models=self.dqn_policy)

        self.save_tensors_to_memory = bool(agent_state['experience_saved_as_tensors'])

        if (len(experience) == 4):
            for s0, a0, r1, s1 in zip(experience[0], experience[1], experience[2], experience[3]):
                if (self.save_tensors_to_memory):
                    self.buffer_memory.insert(self.transition_experience(s0, a0, r1, s1))
                else:
                    if (s1 is not None):
                        self.buffer_memory.insert(self.transition_experience(LazyFrames(None, s0),
                                                                             np.uint8(a0),
                                                                             np.float16(r1),
                                                                             LazyFrames(None, s1)))
                    else:
                        self.buffer_memory.insert(self.transition_experience(LazyFrames(None, s0),
                                                                             np.uint8(a0),
                                                                             np.float16(r1),
                                                                             None))
        del agent_state, experience
