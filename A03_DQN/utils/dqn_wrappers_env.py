"""
DQN-Wrappers-Env

@author: [Luiz Resende Silva](https://github.com/luiz-resende)
@date: Created on Tue Oct 19, 2021
@version: Revised on Wed Dec 01, 2021

This script contains the necessary environment class wrappers to preprocess the
OpenAI-Gym and ALE Atari environments, as well as the MinAtar environment. The base
code was extracted from ``baselines.common.atari_wrappers`` and modified to
clean-up the code and include new methods and classes.

"""
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import gym
from gym import spaces
from gym.wrappers import TimeLimit
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0."""

    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """
        Step observation in the environment.

        Parameters
        ----------
        ac : int
            Action taken.

        Returns
        -------
        numpy.ndarray
            Next state given action chosen.
        """
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""

    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        """
        Reset environment to initial state.

        Returns
        -------
        obs : numpy.ndarray
            Initial state observation.
        """
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """
        Step observation in the environment.

        Parameters
        ----------
        ac : int
            Action taken.

        Returns
        -------
        numpy.ndarray
            Next state given action chosen.
        """
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.

    Done by DeepMind for the DQN and co. since it helps value estimation.

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        Environment.
    """

    def __init__(self, env=None):
        super(EpisodicLifeEnv, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        """
        Step observation in the environment.

        Parameters
        ----------
        action : int
            Action to be taken.

        Returns
        -------
        numpy.ndarray
            Next state given action chosen.
        """
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # Check current lives, make loss of life terminal then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if ((lives < self.lives) and (lives > 0)):
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if (self.was_real_done):
            obs = self.env.reset(**kwargs)
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every 'skip'^{th} frame

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        Environment.
    skip : int, optional
        Number of frames to skip. The default is 4.
    """

    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.

        Parameters
        ----------
        action : int
            Action to be taken.

        Returns
        -------
        max_frame : numpy.ndarray
            Array of frames.
        total_reward : float
            Reward.
        done : bool
            Flag if it is a terminal state.
        info : str
            Extra information.

        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if (i == (self._skip - 2)):
                self._obs_buffer[0] = obs
            if (i == (self._skip - 1)):
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """
        Resets environment to initial state.

        Returns
        -------
        obs : numpy.ndarray
            Initial state observation.
        """
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips rewards and makes them stay inside set {-1, 0, 1}.

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        Atari environment.

    Methods
    -------
    reward()
        Clipped reward.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Reward clipped to the set {+1, 0, -1} by its sign.

        Parameters
        ----------
        reward : float
            Reward true value.

        Returns
        -------
        float
            Reward clipped to interval [-1, 1].
        """
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.

    If the environment uses dictionary observations, ```dict_space_key``` can be specified which indicates which
    observation should be warped.

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        Environment.
    height : int, optional
        Height of warped image. The default is 84.
    width : int, optional
        Width of warped image. The default is 84.
    grayscale : bool, optional
        Flag to whether or not convert frames to black and white. The default is True.
    """

    def __init__(self, env, height=84, width=84, grayscale=True, dict_space_key=None):
        gym.ObservationWrapper.__init__(self, env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if (self._grayscale):
            num_colors = 1  # Black
        else:
            num_colors = 3  # RGB

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
            )
        if (self._key is None):
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert ((original_space.dtype == np.uint8) and (len(original_space.shape) == 3))

    def observation(self, obs):
        """
        Processing observation.

        Parameters
        ----------
        obs : numpy.ndarray
            Environment state observation.

        Returns
        -------
        obs : numpy.ndarray
            Processed environment state observation.
        """
        if (self._key is None):
            frame = obs
        else:
            frame = obs[self._key]

        if (self._grayscale):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if (self._grayscale):
            frame = np.expand_dims(frame, -1)

        if (self._key is None):
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame

        return obs


class FrameStack(gym.Wrapper):
    """
    Stacks n last frames and returns lazy array, which is much more memory efficient.

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        Environment.
    n : int
        Number of frames to stack.

    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.frames = deque([], maxlen=n)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], obs_shape[2] * n),
            dtype=np.uint8)

    def reset(self):
        """
        Reset environment to initial state.

        Return
        ------
        Initial state observation. Array of LazyFrames.
        """
        observation = self.env.reset()
        for _ in range(self.n):
            self.frames.append(observation)
        return self._get_ob()

    def step(self, action):
        """
        Step observation in the environment.

        Parameters
        ----------
        action : int
            Action taken from the available env.action_space.n possible actions.

        Return
        ------
        Observation state given an action from env.action_space.n possible actions.
        """
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        """
        Converts observation.

        Returns
        -------
        dqn_wrappers.LazyFrames
            LazyFrame converted observed state.
        """
        assert len(self.frames) == self.n
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Returns a numpy.array with the frames' pixels scaled within the interval [0.0, 1.0].

    It exists purely to optimize memory usage, which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.

    Parameters
    ----------
    frames : numpy.ndarray
        Array with frames, i.e., state observation.
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=env.observation_space.shape,
            dtype=np.float32)

    def observation(self, observation):
        """
        Method converts LazyFrame to an array and scales its values to [0.0, 1.0]

        Parameters
        ----------
        observation : LazyFrames
            LazyFrames container observation.

        Returns
        -------
        observation : np.ndarray
            Numpy array of floats.

        Notes
        -----
        Careful! This undoes the memory optimization, use with smaller replay buffers only.
        """
        observation = (np.array(observation).astype(np.float32) / 255.0)
        return observation


class LazyFrames(object):
    """
    Object ensures that common frames between the observations are only stored once.

    It exists purely to optimize memory usage, which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.

    Parameters
    ----------
    frames : numpy.ndarray
        Array with frames, i.e., state observation.
    """

    def __init__(self, frames, out=None):
        self._frames = frames
        self._out = out

    def _force(self):
        """
        Concatenating frames.

        Returns
        -------
        numpy.ndarray
            Updated _out argument.
        """
        if (self._out is None):
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        """
        Allow creating and converting custom array container.

        Parameters
        ----------
        dtype : type, optional
            Type of values in the object. The default is None.

        Returns
        -------
        out : numpy.ndarray
            Array of chosen data type.
        """
        out = self._force()
        if (dtype is not None):
            out = out.astype(dtype)
        return out

    def __len__(self
                ) -> int:
        """
        Method to get object's length

        Returns
        -------
        int
            Length of object.
        """
        return len(self._force())

    def __getitem__(self,
                    i: int
                    ) -> Any:
        """
        Method get object item.

        Parameters
        ----------
        i : int
            Index of item.

        Returns
        -------
        Any
            Item at the ith position.
        """
        return self._force()[i]

    def count(self
              ) -> int:
        """
        Gets the number of frames stacked.

        Returns
        -------
        int
            The number of frames in the container.
        """
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self,
              i: int
              ) -> np.ndarray:
        """
        Method gets the ith stack frame.

        Parameters
        ----------
        i : int
            The ith frame to get.

        Returns
        -------
        np.ndarray
            Array of shape (width, heigh) and data type np.uint8.
        """
        return self._force()[..., i]  # The three dots are used to skip to the last dimension.

    def tolist(self,
               dtype: Optional[Union[type, str, None]] = None
               ) -> List[Any]:
        """
        Allow creating and converting custom array container.

        Parameters
        ----------
        dtype : Union[type, str, None], optional
            Type of values in the final list object. It will eventually follow the
            data types accepted by python lists. If ``dtype=None``, it will maintain
            the current data type the LazyFrames object has. The default is None.

        Returns
        -------
        out : numpy.ndarray
            Array of chosen data type.
        """
        out = self._force()
        if (dtype is not None):
            out = out.astype(dtype)
        return out.tolist()


class MinAtarEnvRGB(gym.ObservationWrapper, gym.Wrapper, gym.Env):
    """
    A Gym wrapper around the BaseEnv for MinAtar environment.

    This wrapper converts the n-channel states to RGB. It also allows frame rescaling
    and rendering in ``render_mode='human'`` (as well as ``gym.wrappers.Monitor``).
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'array']}

    def __init__(self,
                 env: gym.envs,
                 frame_width: Optional[int] = 10,
                 frame_height: Optional[int] = 10,
                 grayscale: Optional[bool] = False) -> None:
        super(MinAtarEnvRGB, self).__init__(env)
        self._action_set = self.action_set
        self._action_space = spaces.Discrete(len(self._action_set))
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._grayscale = grayscale
        if (self._grayscale):
            num_colors = 1  # Black
        else:
            num_colors = 3  # RGB
        new_space = gym.spaces.Box(
            high=255,
            low=0,
            shape=(self._frame_width, self._frame_height, num_colors),
            dtype=np.uint8,
            )
        original_space = self.observation_space
        self._obs_space = new_space
        assert ((original_space.dtype == bool) and (len(original_space.shape) == 3))
        self.viewer = None

    def __getScreenRGB(self,
                       original_state: np.ndarray
                       ) -> np.ndarray:
        """
        Method converts the original' environment observation from a 4 channels array to a RGB image.

        Parameters
        ----------
        original_state : np.ndarray
            Original observation in the ```observation_space.shape=(10, 10, 4)```.

        Returns
        -------
        new_atate_rgb : np.ndarray
            Converted observation in the ```observation_space.shape=(10, 10, 3)```, i.e., RGB image.

        Notes
        -----
        The original observation is composed of an array of shape (10, 10, n), where each of the n
        channels corresponds to a game-specific object, e.g., ball, paddle and brick in the
        game ```Breakout-v1```.
        """
        channel_to_rgb = {0: [255, 0, 0],
                          1: [0, 255, 0],
                          2: [0, 0, 255],
                          3: [128, 128, 0],
                          4: [128, 0, 128],
                          5: [0, 128, 128],
                          6: [170, 170, 85],
                          7: [170, 85, 170],
                          8: [85, 170, 170],
                          9: [85, 85, 170]
                          }
        original_shape = original_state.shape
        new_shape_rgb = (original_shape[0], original_shape[1], 3)
        new_state_rgb = np.zeros(new_shape_rgb, dtype=np.uint8)
        for x in range(original_shape[0]):
            for y in range(original_shape[1]):
                for z in range(original_shape[2]):
                    if (original_state[x, y, z]):
                        new_state_rgb[x, y] = np.array(channel_to_rgb[z], dtype=np.uint8)
        return new_state_rgb

    def __observation(self,
                      obs: np.ndarray
                      ) -> np.ndarray:
        """
        Processing observation.

        Parameters
        ----------
        obs : numpy.ndarray
            Environment state observation.

        Returns
        -------
        obs : numpy.ndarray
            Processed environment state observation.
        """
        obs = self.__getScreenRGB(obs)
        if (self._grayscale):
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)
        if (self._grayscale):
            obs = np.expand_dims(obs, -1)

        return obs

    def get_action_meanings(self
                            ) -> List[str]:
        """Return the meaning of each integer action from the set of available actions."""
        meanings = {'n': 'NOOP',
                    'l': 'LEFT',
                    'u': 'UP',
                    'r': 'RIGHT',
                    'd': 'DOWN',
                    'f': 'FIRE'}
        env_action_map = self.unwrapped.game.env.action_map
        actions = [meanings[env_action_map[a]] for a in self.unwrapped.action_set]
        return actions

    def step(self,
             action: int
             ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Perform one agent step.

        Parameters
        ----------
        action: int
            Action index to selected to be executed.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            State transitino tuple with the (observation, reward, terminal, metadata).

        Notes
        -----
        ```metadata``` is empty since the environment does not keep track of lives.
        """
        action = self.action_set[action]
        reward, done = self.game.act(action)
        metadata = {}
        obs = self.unwrapped.game.state()

        return (self.__observation(obs), reward, done, metadata)

    def reset(self
              ) -> np.ndarray:
        """Resets environment and returns initial observation."""
        self.unwrapped.game.reset()
        obs = self.unwrapped.game.state()
        obs = self.__observation(obs)

        return obs

    def render(self,
               mode: str
               ) -> Union[np.ndarray, Any]:
        """
        Method to render the environment.

        Parameters
        ----------
        mode : str
            The rendering mode.

        Returns
        -------
        Union[np.ndarray, Any]
            Either the array with the state observation or a gym-based image viewer.
        """
        if mode == "array":

            return self.unwrapped.game.state()
        elif mode == 'rgb_array':
            img = self.__observation(self.unwrapped.game.state())

            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering
            img = self.__observation(self.unwrapped.game.state())
            if (self.viewer is None):
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

            return self.viewer.isopen

    def close(self
              ) -> None:
        """Cleanup any leftovers by the environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def action_space(self
                     ) -> spaces.Discrete:
        """Return Gym's action space."""
        return self._action_space

    @property
    def observation_space(self
                          ) -> spaces.Box:
        """Return Gym's action space."""
        return self._obs_space


def make_atari_env(game_id, render_mode='rgb_array', max_episode_steps=None, no_op_reset=True, no_op_max=30,
                   skip_frames=True, skip_frames_n=4):
    """
    Method makes an OpenAI Gym Atari environment.

    Parameters
    ----------
    game_id : str
        The Atari environment name.
    render_mode : str, optional
        The render mode for the environment. The default is 'rgb_array'.
    max_episode_steps : Union[int, None], optional
        The maximum number of steps an episode can have. The default is None.
    no_op_reset : bool, optional
        Flag to whether or not have number of no operation actions. The default is True.
    no_op_max : int, optional
        Maximum number of no-operations. The default is 30.
    skip_frames : bool, optional
        Flag to whether or not skip frames. The default is True.
    skip_frames_n : int, optional
        Number of frames to skip. The default is 4.

    Returns
    -------
    env : Any
        A wrappend gym.envs.atari.environment.AtariEnv.
    """
    env = gym.make(game_id, render_mode=render_mode).env
    if (no_op_reset):
        env = NoopResetEnv(env, noop_max=no_op_max)
    if (('NoFrameskip' in env.spec.id) and skip_frames):
        env = MaxAndSkipEnv(env, skip=skip_frames_n)
    if (max_episode_steps is not None):
        env = TimeLimit(env, max_episode_steps=int(max_episode_steps))

    return env


def wrap_atari_env(env, clip_rewards=True, episodic_life=True, scale_frame=False, stack_frames=False, stack_frames_n=4,
                   warp_frames=True, warp_frames_greyscale=True, warp_frames_size=(84, 84)):
    """
    Configure environment for DeepMind-style Atari environment from Mnih et al. (2015).

    Parameters
    ----------
    env : gym.envs.atari.environment.AtariEnv
        The Atari OpenAI Gym environment from ALE. Can be a wrapped environment.
    clip_rewards : bool, optional
        Flag to whether or not to clip the reward to the interval [-1, 1]. The default is True.
    episodic_life : bool, optional
        Flag to whether or not it is an episodic life environment. The default is True.
    scale_frame : bool, optional
        Flag to whether or not normalize the frame values, i.e., [0, 255] -> [0, 1]. The default is False.
    stack_frames : bool, optional
        Flag to whether or not stack the frames. The default is True.
    stack_frames_n : int, optional
        Number of frames to stack. The default is 4.
    warp_frames : bool, optional
        Flag to whether or not resample and crop frames. The default is True.
    warp_frames_greyscale : TYPE, optional
        Flag to whether or not convert frames to greyscale. The default is True.
    warp_frames_size : Tuple, optional
        The output shape (height, width) for the warped frames. The default is (84, 84).

    Returns
    -------
    env : Any
        The environment after all preprocessing steps selected.
    """
    if (episodic_life):
        env = EpisodicLifeEnv(env)
    if ('FIRE' in env.unwrapped.get_action_meanings()):
        env = FireResetEnv(env)
    if (warp_frames):
        env = WarpFrame(env, height=warp_frames_size[0], width=warp_frames_size[1], grayscale=warp_frames_greyscale)
    if (scale_frame):
        env = ScaledFloatFrame(env)
    if (clip_rewards):
        env = ClipRewardEnv(env)
    if (stack_frames):
        env = FrameStack(env, n=stack_frames_n)

    return env


def make_minAtar_env(game_id, render_mode='rgb_array'):
    """
    Method makes an OpenAI Gym MinAtar environment.

    Parameters
    ----------
    game_id : str
        The MinAtar game environment name.
    render_mode : str, optional
        The render mode for the environment. The default is 'rgb_array'.

    Returns
    -------
    env : gym.BaseEnv
        The MinAtar game wrapped as an OpenAI Gym environment.
    """
    env = gym.make(game_id, render_mode=render_mode).env

    return env
