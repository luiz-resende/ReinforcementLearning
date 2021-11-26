# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:59:20 2021

@author: Luiz Resende Silva
"""
from typing import Any, Sequence, Tuple, Optional, Union
import numpy as np
import collections

import gym
import tqdm
import time
# import wandb

import torch
from dqn_memory_buffer import MemoryBuffer
from dqn_models_torch import DQNModel  # ModelDQN
from dqn_wrappers_env import make_atari_env, wrap_atari_env


class AgentDQN():

    def __init__(
            self,
            game_id: Optional[str] = 'Pong-v0',  # Name of the game environment to build
            render_mode: Optional[str] = 'rgb_array',  # Render mode for building trining environment
                                                       # ```render_mode='human'``` slows training process.
            normalize_frames: Optional[bool] = True,  # Whether or not to scale frames from [0, 255] -> [0.0, 1.0]
            max_episode_steps_env: Optional[Union[int, None]] = None,  # Timelimit wrapper
            no_op_reset_env: Optional[bool] = True,  # Whether or not use no-op wrapper
            no_op_max_env: Optional[int] = 30,  # Maximum number of no-op actions
            skip_frames_env: Optional[bool] = True,  # Whether or not to skip a given number of frames
            skip_frames_env_n: Optional[int] = 4,  # Number of frames to skip
            memory_size: Optional[int] = 1000000,  # Maximum size of the replay memory from which sample experience
            batch_size: Optional[int] = 32,  # Size of experience batch sampled from replay memory
            replay_start_size: Optional[int] = 50000,  # Initial number of actions taken using uniform random
                                                       # policy to build initial experience
            gamma_disc: Optional[float] = 0.99,  # Discounting factor gamma
            alpha_lr: Optional[float] = 0.00025,  # Learning rate alpha used by optimizer
            epsilon_max: Optional[float] = 1.0,  # Initial/maximum exploration probability
            epsilon_min: Optional[float] = 0.1,  # Final/minimum exploration probability
            epsilon_decay: Optional[int] = 1000000,  # The number of episodes in which the exploration probability
                                                     # will decrease linearly
            target_network_update: Optional[int] = 10000,  # Number of time steps after which to update the
                                                           # target network
            grad_momentum: Optional[float] = 0.95,  # Gradient momentum used by optimizer
            grad_momentum_square: Optional[float] = 0.95,  # Squared gradient momentum used by optimizer.
            min_sqr_grad: Optional[float] = 0.01,  # Constant added to the squared gradient in the denominator
                                                   # of the optimizer for stability
            max_number_training_frames: Optional[int] = 50000000,  # The hard stop based on number of training timestep
                                                                   # iterations (frames generated)
            loss_criterion: Optional[Any] = None,  # The criterion to calculate the loss
            device_agent: Optional[str] = 'cpu',  # The device where to move torch tensors and models
            seed: Optional[int] = 895359  # Seed used for reproducibility
            ) -> None:
        # SAVING SEED AND CREATING RANDOM NUMBER GENERATOR
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)
        # CREATING THE ATARI ENVIRONMENT
        if ('NoFrameskip' in game_id):
            self.game_id = game_id
        else:  # Ensuring the environment is not skiping frames already
            self.game_id = game_id[:-3] + 'NoFrameskip' + game_id[-3:]
        self.render_mode = render_mode
        self.env = make_atari_env(self.game_id, render_mode=self.render_mode, max_episode_steps=max_episode_steps_env,
                                  no_op_reset=no_op_reset_env, no_op_max=no_op_max_env, skip_frames=skip_frames_env,
                                  skip_frames_n=skip_frames_env_n)  # Creating Atari ALE environment
        self.action_space_n = self.env.action_space.n
        self.normalize_frames = normalize_frames
        time_init = time.strftime('%Y-%m-%d_%Hh%M', time.localtime())
        self.video_direc = (r'./videos/dqn_video_%s_%s' % (self.game_id[:-3].lower(), time_init))
        # SETTING SEEDS
        self.env.seed(self.__seed)
        np.random.seed(self.__seed)
        torch.manual_seed(self.__seed)
        # SETTING TORCH DEVICE
        self.device = torch.device('cpu')
        if (((device_agent.lower() == 'gpu') or (device_agent.lower() == 'cuda')) and torch.cuda.is_available()):
            self.device = torch.device('cuda')
        # SETTING REPLAY MEMORY PARAMETERS
        self.replay_memory = MemoryBuffer(max_size=memory_size, seed=self.__seed)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        # SETTING OPTIMIZER PARAMETERS
        self.gamma_disc = gamma_disc
        self.alpha_lr = alpha_lr
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = ((self.epsilon_max - self.epsilon_min) / epsilon_decay)
        self.grad_momentum = grad_momentum
        self.grad_momentum_square = grad_momentum_square
        self.min_sqr_grad = min_sqr_grad
        self.max_number_training_frames = max_number_training_frames
        self.loss_criterion = loss_criterion
        if (self.loss_criterion is None):
            self.loss_criterion = torch.nn.SmoothL1Loss().to(self.device)  # Setting Huber Loss as default if None is passed
        else:
            self.loss_criterion = self.loss_criterion.to(self.device)
        # SETTING POLICY AND TARGET MODEL PARAMETERS
        self.target_network_update = target_network_update
        self.dqn_policy = None
        self.dqn_target = None
        self.optimizer = None
        # SETTING CONTAINERS FOR STORING TRAINING METRICS
        self.episodes_scores = [[], []]
        self.episodes_losses = [[], []]
        self.episodes_scores_eval = [[], []]
        self.total_step_count = 0

    def seed(self, seed: Optional[Union[int, None]] = None):
        """
        Method sets the global seed and random number generator.

        Parameters
        ----------
        seed : Union[int, None], optional
            Seed for random number generator. The default is None.

        Returns
        -------
        int
            Seed value if ```seed=None```.
        """
        if (seed is None):
            return self.__seed
        else:
            if (isinstance(seed, int) or isinstance(seed, int)):
                self.__seed = int(seed)
                self.__rng = np.random.default_rng(self.__seed)
                self.env.seed(self.__seed)
                np.random.seed(self.__seed)
                torch.manual_seed(self.__seed)
            else:
                raise TypeError("TypeError! seed should be integer, instead got %s..." % str(type(seed)))

    def build_q_networks(self, wrap_env=True, clip_rewards=True, episodic_life=True, scale_frame=False,
                         stack_frames=True, warp_frames=True, warp_frames_greyscale=True, **model_kwargs) -> None:
        """
        Method sets both the policy and the target q-networks.

        Creates agents using the default arguments {'in_channels': 4, 'out_channel': 16, 'shape_input': (84, 84),
                                                    'kernel': (8, 8), 'stride': (4, 4), 'padding': (0, 0),
                                                    'out_features_linear': 256, 'number_actions': gym.envs.action_space.n,
                                                    'agent_architecture': 1},
        which sums to the model proposed in
        Mnih et al. (2013). These values are changed with the argument passed in the function.

        Parameters
        ----------
        wrap_env : bool, optional
            Whether or not to wrap environment. The default is True.
        clip_rewards : bool, optional
            Whether or not to clip de rewards to [-1, 1]. The default is True.
        episodic_life : bool, optional
            Whether or not the environment has episodic life. The default is True.
        scale_frame : bool, optional
            Whether or not to scale frame pixels in the interval [0.0, 1.0]. The default is False.
        stack_frames : bool, optional
            Whether or not to stack a given number of frames. The default is True.
        warp_frames : bool, optional
            Whether or not to warp the frames from their original size to a new shape. The default is True.
        warp_frames_greyscale : bool, optional
            Whether or not to make frames to greyscale. The default is True.
        **model_kwargs : Any, optional
            Arbitrary keyword arguments contained in {'in_channels', 'out_channel', 'shape_input', 'kernel', 'stride',
            'padding', 'out_features_linear', 'number_actions', 'agent_architecture'} used to build network architecture.
            If none is passed, the default values are used (Mnih et al., 2013).
        """
        show_args = False
        build_args = {'in_channels': self.env.observation_space.shape[-1], 'out_channel': 16, 'shape_input': (84, 84),
                      'kernel': (8, 8), 'stride': (4, 4), 'padding': (0, 0), 'out_features_linear': 256,
                      'number_actions': self.action_space_n, 'agent_architecture': 1}
        for key in model_kwargs.keys():
            if (key in list(build_args.keys())):
                build_args[key] = model_kwargs[key]
            else:
                show_args = True

        if (show_args):
            print("Available parameters to change: ", list(build_args.keys()))

        self.dqn_policy = DQNModel(in_channels=build_args['in_channels'],
                                   out_channel=build_args['out_channel'],
                                   shape_input=build_args['shape_input'],
                                   kernel=build_args['kernel'],
                                   stride=build_args['stride'],
                                   padding=build_args['padding'],
                                   out_features_linear=build_args['out_features_linear'],
                                   number_actions=build_args['number_actions'],
                                   agent_architecture=build_args['agent_architecture'])
        self.dqn_policy.to(self.device)

        self.dqn_target = DQNModel(in_channels=build_args['in_channels'],
                                   out_channel=build_args['out_channel'],
                                   shape_input=build_args['shape_input'],
                                   kernel=build_args['kernel'],
                                   stride=build_args['stride'],
                                   padding=build_args['padding'],
                                   out_features_linear=build_args['out_features_linear'],
                                   number_actions=build_args['number_actions'],
                                   agent_architecture=build_args['agent_architecture'])
        self.dqn_target.to(self.device)

        if (wrap_env):
            self.env = wrap_atari_env(self.env, clip_rewards=clip_rewards, episodic_life=episodic_life,
                                      scale_frame=scale_frame, stack_frames=stack_frames,
                                      stack_frames_n=build_args['in_channels'], warp_frames=warp_frames,
                                      warp_frames_greyscale=warp_frames_greyscale,
                                      warp_frames_size=build_args['shape_input'])

        self.env_monitor = gym.wrappers.Monitor(self.env, directory=self.video_direc, force=True,
                                                video_callable=lambda episode: True)

    def frame_tensor(self, obs) -> torch.Tensor:
        """
        Method converts observation state frame from atari_wrappers.LazyFrames to torch.Tensor.

        Parameters
        ----------
        obs : dqn_wrappers.LazyFrames
            Atari frames preprocessed to needed shapes and number of channels. Frames are kept as LazyFrames objects to
            improve memory efficiency. This method then quickly converts the frames to torch.Tensor of shapes
            torch.Size([1, number_channels, frame_height, frame_width]) when they are required to be used in the model.

        Returns
        -------
        torch.Tensor
            State frame converted to torch.Tensor of torch.Size([1, number_channels, frame_height, frame_width]).
        """
        obs = np.array(obs, dtype=np.float)
        obs = obs.transpose((2, 0, 1))  # Transposing matrix to have number channels first
        if (self.normalize_frames):
            obs = (obs / 255.0)  # Scaling pixel values between 0.0 and 1.0
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        return obs.unsqueeze(0)

    def sample_action(self, obs):
        """
        Method selects action using e-greedy.

        Parameters
        ----------
        obs : dqn_wrappers.LazyFrames
            Atari frames preprocessed to match deepmind implementation. The frames are kept as LazyFrames objects in the
            buffer memory container for memory efficiency. They are converted to torch.Tensor when selecting the action.

        Returns
        -------
        act : np.uint8
            Integer of type uint8 with selected action from env.action_space. Converted to uint8 for memory efficiency.
        """
        e_prob = self.__rng.random()
        if ((e_prob < self.epsilon_max) or (self.replay_memory.size < self.replay_start_size)):
            act = self.__rng.choice(self.action_space_n)
            act = np.uint8(act)
        else:
            with torch.no_grad():
                act = self.dqn_policy.forward(self.frame_tensor(obs)).max(1)[1].to('cpu')
                act = np.uint8(act.item())
        return act

    def get_batch_replay(self, batch_samples: Sequence) -> Tuple:
        """
        Method retrieves the list of tuples from buffer memory and groups them in specific batches.

        Groups using collections.NamedTuple. It then converts all the values in the mini batch to torch.Tensors, moving
        them to the device set (i.e., cpu or cuda). States, actions, rewards and other information are not saved in the
        buffer memory as tensors to be memory efficient, converting only the mini batch to tensors when using it.

        Parameters
        ----------
        batch_samples : Sequence
            List of tuples (state_{t}, action_{t}, reward_{t+1}, state_{t+1}, isdone_{t+1}).

        Returns
        -------
        (batch_s_tp0, batch_a_tp0, batch_r_tp1, batch_s_tp1, batch_d_tp1) : Tuple
            s_tp0 : torch.Tensor
                Tensor of torch.Size([batch_size, number_channels, frame_height, frame_width])
                for observed statesat time {t}.
            a_tp0 : torch.Tensor
                Tensor of torch.Size([batch_size, 1]) for observed action taken at time {t}.
            r_tp1 : torch.Tensor
                Tensor of torch.Size([batch_size]) for observed reward received at time {t+1}.
            s_tp1 : torch.Tensor
                Tensor of torch.Size([~batch_size, number_channels, frame_height, frame_width])
                for observed states at {t+1}.
            d_tp1 : torch.Tensor
                Tensor of torch.Size([batch_size]) mapping which state frames at {t+1} are terminal
                states (torch.bool=True) and which are not (torch.bool=False).
        """
        observations = collections.namedtuple('transition_samples', ('obs_tp0', 'act_tp0', 'rew_tp1', 'obs_tp1', 'done_tp1'))

        batch_samples = observations(*zip(*batch_samples))

        # Grouping current states and converting to tensors - Needs to be torch.float32 or torch.float
        # states = tuple((map(lambda s: self.frame_tensor(s), batch_samples.state)))
        batch_s_tp0 = [self.frame_tensor(obs0) for obs0 in batch_samples.obs_tp0]
        batch_s_tp0 = torch.cat(batch_s_tp0)
        # Grouping actions - Needs to be torch.long or torch.int64
        # actions = tuple((map(lambda a: torch.tensor([[a]], dtype=torch.int32, device=self.device), batch_samples.action)))
        batch_a_tp0 = [torch.tensor([[a]], dtype=torch.long, device=self.device) for a in batch_samples.act_tp0]
        batch_a_tp0 = torch.cat(batch_a_tp0)
        # Grouping rewards - Data type torch.float or torch.float32
        # rewards = tuple((map(lambda r: torch.tensor([r], dtype=torch.float32, device=self.device), batch_samples.reward)))
        batch_r_tp1 = [torch.tensor([r], dtype=torch.float, device=self.device) for r in batch_samples.rew_tp1]
        batch_r_tp1 = torch.cat(batch_r_tp1)
        # Grouping next states and converting to tensors - Need to be torch.float32 or torch.float
        # batch_s_tp1 = [self.frame_tensor(obs1) for obs1 in batch_samples.obs_tp1 if obs1 is not None]
        batch_s_tp1 = [self.frame_tensor(obs1) for obs1 in batch_samples.obs_tp1]
        batch_s_tp1 = torch.cat(batch_s_tp1)
        # Grouping is_terminal Boolean flags and converting to tensors - Need to be torch.bool
        # Its inverse is used as masks of non-final states
        # is_done = tuple((map(lambda d: torch.tensor([d], dtype=torch.bool, device=self.device), batch_samples.done)))
        batch_d_tp1 = [torch.tensor([d], dtype=torch.bool, device=self.device) for d in batch_samples.done_tp1]
        batch_d_tp1 = torch.cat(batch_d_tp1)

        return (batch_s_tp0, batch_a_tp0, batch_r_tp1, batch_s_tp1, batch_d_tp1)

    def optimize_model(self) -> None:
        """
        Method optimizes deep q-network model.

        Parameters
        ----------
        None

        Returns
        -------
        loss : float
            The loss calculated from the current optimization step.

        Notes
        -----
        The policy and target agents' parameters are optimized after each iteration by calling this method.
        """
        b_s_t0, b_a_t0, b_r_t1, b_s_t1, b_done_t1 = self.get_batch_replay(self.replay_memory.random_samples(self.batch_size))

        # Computing the Q(s_{t}, a) action values, actions which taken for each batch state according to dqn_policy
        q_vals_policy = self.dqn_policy.forward(b_s_t0).gather(1, b_a_t0)

        # Initializing next state action values to 0, guaranteeing the expected state value
        # is 0 in terminal states (as it should have).
        # q_vals_st1 = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        q_vals_st1 = self.dqn_target.forward(b_s_t1).max(1)[0].detach()
        q_vals_st1[b_done_t1] = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # Calculating V(s_{t+1}) for all next states, and selecting the best reward with max(1)[0].
        # Merging based on the mask, having either the expected state value or 0 if the state is final.
        # q_vals_st1[b_done_t1.logical_not()] = self.dqn_target.forward(b_s_t1).max(1)[0].detach()
        # q_vals_st1[next_s_masks] = self.dqn_target.forward(batch_next_s).max(1)[0].detach()
        # Calculating expected values for next states
        q_vals_expect = (b_r_t1 + (self.gamma_disc * q_vals_st1))

        # Calculating Loss
        loss = self.loss_criterion(q_vals_policy, q_vals_expect.unsqueeze(1))

        # Optimizing model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.detach().item()

    def train_agent(self,
                    optim: Optional[str] = 'RMSprop',
                    max_number_episodes: Optional[int] = 1000000,
                    max_number_timestep: Optional[int] = 10000,
                    update_epsilon_step: Optional[bool] = True,
                    normalize_frames: Optional[bool] = True,
                    render: Optional[bool] = False,
                    render_mode: Optional[str] = 'human',
                    continue_training: Optional[bool] = False
                    ):
        """
        Method to call agent's pipeline training.

        Parameters
        ----------
        optim : str, optional
            The optimizer to be used. The implemented training pipeline gives the user the option to choose on of the
            following optimizers: 'RMSprop', 'Adam', 'Adadelta', and 'SGD'. Any of which instantiate a criteria using
            the hyperparameters passed when instantiating the pipeline class. The default is 'RMSprop'.
        max_number_episodes : int, optional
            Number of episodes to generate. The default is 1000000.
        max_number_timestep : int, optional
            Maximum number of time steps within an episode. The default is 10000.
        update_epsilon_step : bool, optional
            Whether to decrease epsilon at each time step (True), or at each episode (False). The default is True.
        normalize_frames : bool, optional
            Whether or not to scale pixel values from [0, 255] -> [0.0, 1.0] in the frames. The default is True.
        render : bool, optional
            Whether or not to render the episode while training. The default is False.
        render_mode : str, optional
            Episode rendering mode if ```render=True```. The default is 'human'.
        continue_training : bool, optional
            Whether or not the agent's training should start from scratch. The default is False:

        Returns
        -------
        dqn_models_torch
            Returns the trained policy Q-network.
        """
        if (not continue_training):
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.normalize_frames = normalize_frames

        try:
            if (optim.lower() == 'adam'):
                self.optimizer = torch.optim.Adam(self.dqn_policy.parameters(),
                                                  lr=self.alpha_lr,
                                                  eps=self.min_sqr_grad)
            elif (optim.lower() == 'sgd'):
                self.optimizer = torch.optim.Adam(self.dqn_policy.parameters(),
                                                  lr=self.alpha_lr,
                                                  momentum=self.grad_momentum)
            elif (optim.lower() == 'adadelta'):
                self.optimizer = torch.optim.RMSprop(self.dqn_policy.parameters(),
                                                     rho=self.grad_momentum_square,
                                                     lr=self.alpha_lr,
                                                     eps=self.min_sqr_grad)
            else:  # Uses RMSprop by default
                self.optimizer = torch.optim.RMSprop(self.dqn_policy.parameters(),
                                                     lr=self.alpha_lr,
                                                     momentum=self.grad_momentum,
                                                     alpha=self.grad_momentum_square,
                                                     eps=self.min_sqr_grad)

            ep_progress_bar = tqdm.tqdm(range(max_number_episodes), desc=r'[Training AgentDQN] ', unit=' ep')
            for e in ep_progress_bar:
                time.sleep(0.0)
                ep_score = []
                ep_loss = []

                s_t0 = self.env.reset()
                if (render):
                    self.env.render(mode=render_mode)

                ep_timesteps = 0
                for t in range(max_number_timestep):
                    self.total_step_count += 1
                    ep_timesteps += 1

                    a_t0 = self.sample_action(s_t0)
                    s_t1, r_t1, done, _ = self.env.step(a_t0)
                    ep_score.append(r_t1)  # Saving current plus score
                    self.replay_memory.add((s_t0, a_t0, r_t1, s_t1, done))  # Adding transition tuple to memory
                    s_t0 = s_t1

                    if (self.replay_memory.size >= self.replay_start_size):  # Optimizing after minimum memory achieved
                        e_loss = self.optimize_model()
                        ep_loss.append(e_loss)

                        if (self.total_step_count % self.target_network_update == 0):  # Updating target network
                            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

                    if (update_epsilon_step):  # Epsilon decreases linearly per time step
                        self.epsilon_max = np.max([self.epsilon_min, (self.epsilon_max - self.epsilon_dec)])

                    if (done):
                        break

                self.episodes_scores[0].append(np.sum(ep_score))
                self.episodes_scores[1].append(np.mean(ep_score))

                if (self.replay_memory.size >= self.replay_start_size):
                    self.episodes_losses[0].append(np.sum(ep_loss))
                    self.episodes_losses[1].append(np.mean(ep_loss))

                if (not update_epsilon_step):  # Epsilon decreases linearly per episode
                    self.epsilon_max = np.max([self.epsilon_min, (self.epsilon_max - self.epsilon_dec)])

                if (len(ep_loss) > 0):
                    # wandb.log({"Average loss": np.round(np.mean(self.episodes_losses[1]), decimals=3)})
                    ep_progress_bar.set_postfix(AvgSteps=np.round((self.total_step_count / (e + 1)), decimals=3),
                                                AvgLoss=np.round(np.mean(self.episodes_losses[1]), decimals=3),
                                                RewardMax=np.max(self.episodes_scores[0]),
                                                RewardEps=np.sum(ep_score),
                                                TotalSteps=self.total_step_count,
                                                )
                else:
                    ep_progress_bar.set_postfix(AvgSteps=np.round((self.total_step_count / (e + 1)), decimals=3),
                                                AvgLoss='NotOptm',
                                                RewardMax=np.max(self.episodes_scores[0]),
                                                RewardEps=np.sum(ep_score),
                                                TotalSteps=self.total_step_count,
                                                )

                if (self.total_step_count >= self.max_number_training_frames):
                    break  # Stop training if a maximum number of frames has been generated

            self.env.close()
            return self.dqn_policy
        except KeyboardInterrupt:
            print("Saving model before quitting...")
            self.save_dqn_agents(sufix=('InterruptedApprox_%dkIter' % int(self.total_step_count / 1000)))
            self.env.close()
            return self.dqn_policy

    def evaluate_agent(self,
                       number_episodes: Optional[int] = 50,
                       render: Optional[bool] = True,
                       render_mode: Optional[str] = 'human'
                       ) -> None:
        """
        Method to evaluate trained agent.

        Parameters
        ----------
        number_episodes : int, optional
            Number of episode to test the agent. The default is 1000.
        render : bool, optional
            Flag to whether or not render episode. The default is True.
        render_mode : str, optional
            Episode rendering mode. The default is 'human'.

        Returns
        -------
        None.
        """
        score_episode_eval = []

        for e in range(number_episodes):
            obs = self.env_monitor.reset()
            obs = self.frame_tensor(obs)
            done = False

            while (not done):
                action = self.dqn_policy.forward(obs).max(1)[1].item()
                obs, reward, done, _ = self.env_monitor.step(action)
                obs = self.frame_tensor(obs)
                score_episode_eval.append(reward)

                if (render):
                    self.env_monitor.render(mode=render_mode)
                    time.sleep(0.05)

                if (done):
                    break

            if ((e > 0) and ((e + 1) % 5 == 0)):
                episode = int((e + 1) / 5)
                print(f"Finished Episode {episode} with reward {np.sum(score_episode_eval)}")
                if (e < (number_episodes - 1)):
                    score_episode_eval = []

            self.episodes_scores_eval[0].append(np.sum(score_episode_eval))
            self.episodes_scores_eval[1].append(np.mean(score_episode_eval))

        self.env_monitor.close()
        # show_video(self.video_direc)

    def save_dqn_agents(self, path: Optional[str] = r'./saved_models/', sufix: Optional[str] = '') -> None:
        """
        Method calls ```torch.save()``` and saves both the policy and target trained networks to files in a chosen folder.

        Parameters
        ----------
        path : str, optional
            Path of directory to save models. The default is r'./saved_models/'.
        sufix : str, optional
            A sufix string to be appended to the end of the file name. The default is ''.
        """
        base_name = (path + time.strftime('%Y-%m-%d_%Hh%M', time.localtime()))
        policy_name = (base_name + '_policy_q_net__' + sufix + '.pt')
        target_name = (base_name + '_target_q_net__' + sufix + '.pt')
        torch.save(self.dqn_policy, policy_name)
        torch.save(self.dqn_target, target_name)

    def load_dqn_agents(self, path_name: Optional[str] = r'./saved_models/', sufix: Optional[str] = '',
                        load_target: Optional[bool] = False) -> None:
        """
        Method calls ```torch.load()``` and loads both the policy and target trained networks from files in a chosen folder.

        Parameters
        ----------
        path_name : str, optional
            Path of directory to load models. The default is r'./saved_models/'.
        sufix : str, optional
            A sufix string to be appended to the end of the file name. The default is ''.
        load_target : bool, optional
            Wheter or not to load target model. The default is False.

        Notes
        -----
        Method assumes saved models has the model names given by ```saved_dqn_agents()```.
        """
        policy_name = (path_name + '_policy_q_net__' + sufix + '.pt')
        self.dqn_policy = torch.load(policy_name)
        if (load_target):
            target_name = (path_name + '_target_q_net__' + sufix + '.pt')
            self.dqn_target = torch.load(target_name)
