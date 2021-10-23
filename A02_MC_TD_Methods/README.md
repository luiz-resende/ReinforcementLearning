# Monte Carlo Methods and Temporal Difference Learning

Python scripts implementing different Monte Carlo (MC) and Temporal Differnce Learning (TD) reinforcement learning algorithms as shown in *Reinforcement Learning: An Introduction* by [Sutton & Barto (2018)](http://incompleteideas.net/book/RLbook2020.pdf).

# Implementation

The helper methods are designed to generate choose an action $a$ from a set of actions $A(s)$ for a state $s$, $epsilon$-soft policies and greedify them, generate full learning episodes and episodes with Q-table learning. However, it is important to notice that these methods were implemented targeting the *Modified Frozen Lake Environmemnt* ([source code](https://github.com/micklethepickle/modified-frozen-lake)) by Michel Ma, where there are some modifications in how the agent sees and receives rewards when transitioning to terminal states and the maximum number of time steps the episode can have.
