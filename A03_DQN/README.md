# Analysis of Deep Q-Network for Playing Atari

We present an analysis on the Deep Q-Network algorithm present in Mnih et al. (2013, 2015). The tests performed aimed to study the impact that different parameters have in the trained agent's performance.

Among the tests conducted, we analyzed the experience replay memory buffer capacity, the update frequency of the model, the network architecture and the mini-batch size.

## Quick Start

To use the ``AgentDQN`` and its training pipeline, one needs Python3 installed and ensure the ``pip`` is updated[^1]. The model is implemented using ``Pytorch``, hence its installation is also required. All other modules required can be found on the first code cell in the Python notebook with containing the training pipeline

[^1]:  Alternativelly, the training pipeline already has a cell that can be executed to update ``pip``.

## Use and Reproduction

There are four scripts need to train an agent:

1. ``dqn_memory_buffer.py``: contains the class object used for the experience replay memory buffer.
2. ``dqn_wrappers_env.py``: the class wrappers used for the Atari environment preprocessing.
3. ``dqn_models_torch.py``: the neural network model used to approximate $Q$ action values.
4. ``dqn_agent_atari.py``: the DQN agent.

All those are needed for a successful run. The ``atari_dqn_training_pipeline.ipynb`` notebook contains the information used for the agent's training (e.g., model parameters, environment settings, seed, etc.) and following those ensure the reproduction of the results.

## Using on Windows and Linux-based OS

The ``atari_dqn_training_pipeline.ipynb`` was written to be run in Google Colaboratory service, hence its initial commands are target to Python for Linux. Some can still be executed with small modifications (e.g., removing the ``!`` operator in front of the command). However, most do not have Windows equivalents (e.g., ``apt-get`` and ``wget``), and the user must work around this to manage to run the code cells.

The rest of the code is OS-independent and should run smoothly in any system with Python3 and the required modules installed.

## Citing

If you use the available code or the results from it in your research, please cite the following:

Resende Silva, L. Talotsing, G. P. M. (2021). Analysis of Deep Q-Network for Playing Atari. École Polytechnique de Montréal.

In BibTeX format:

```
@techreport{ResendeSilva2021analysis,
author = {{Resende Silva}, Luiz and {Talotsing}, Gaëlle Patricia Megouo},
title = {Analysis of Deep Q-Network for Playing Atari},
year = {2021},
institution = {École Polytechnique de Montréal},
month = {12}
}
```

## References

1. Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The Arcade  LearningEnvironment:  An  Evaluation  Platform  for  General  Agents. *Journal  of Artificial IntelligenceResearch*, 47:253–279, jun  2013. [Available here](https://jair.org/index.php/jair/article/view/10819)

2. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wier-stra, and Martin Riedmiller. Playing Atari with Deep Reinforcement Learning. *arXiv*, pp. 1–9, dec 2013. [Available here](http://arxiv.org/abs/1312.5602.10)

3. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Belle-mare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen,Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wier-stra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learn-ing. *Nature*, 518(7540):529–533, feb 2015. ISSN 0028-0836. doi: 10.1038/nature14236. [Available here](http://dx.doi.org/10.1038/nature14236)

4. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, TrevorKilleen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, EdwardYang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performancedeep learning library. In H. Wallach,  H. Larochelle,  A. Beygelzimer,  F. d'Alch ́e-Buc,E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems 32*, pp.8024–8035. Curran Associates, Inc., 2019. [Available here](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)

5. Kenny Young, and Tian Tian. MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments.  *arXiv preprint arXiv:1903.03176*, 2019. [Avaialble here](https://arxiv.org/abs/1903.03176)

## License

MIT License

Copyright (c) 2021 Luiz Resende Silva

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
