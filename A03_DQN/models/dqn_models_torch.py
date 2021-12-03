# -*- coding: utf-8 -*-
"""
DQN-Model

@author: [Luiz Resende Silva](https://github.com/luiz-resende)
@date: Created on Wed Oct 20, 2021
@version: Revised on Mon Nov 15, 2021

Creates the neural network model used for the Deep Q-Network algorithm.

Revision Notes
--------------
The network parameters were all setup as functions of the construction arguments
to facilitate the modification of the network and the test of different configurations.

"""
from typing import Tuple, Any, Optional, Union
import torch
import collections
import numpy as np
from torchsummary import summary


class DQNModel(torch.nn.Module):  # ModelDQN
    r"""
    Class object implementing a DQN neural network using Pytorch mmodule.

    Creates a deep q-network with either one of the predefined architectures from Mnih et al.(2013)
    or Mnih et al. (2015).

    Parameters
    ----------
    in_channels : ``int``, optional
        Number of input channels (or number of 1-channel stacked frames). The default is 3 (RBG image).
    out_channel : ``int``, optional
        Number of input channels from first convolutional layer. The default is 16.
    shape_input : ``Union[int, Tuple]``, optional
        The shape of input frames. The default is (84, 84).
    kernel : ``Union[int, Tuple]``, optional
        The size of kernel in the first convolutional layer. The default is (8, 8).
    stride : ``Union[int, Tuple]``, optional
        The size of stride in the first convolutional layer. The default is (4, 4).
    padding : ``Union[int, Tuple]``, optional
        The size of padding in the first convolutional layer. The default is (0, 0).
    out_features_linear : ``int``, optional
        Number of output features in the first linear layer. The default is 256.
    number_actions : ``int``, optional
        Number of actions in the 'env.action_space.n'. The default is 4.
    agent_architecture : ``int``, optional
        The type of architecture, 1 for two convolutional layers (Mnih et al., 2013) or 2 for three convolutional layers
        (Mnih et al., 2015). The default is 1.
    use_batch_norm : ``bool``, optional
        Whether or not use batch normalization between layers. The default is ``False``.
    scale_batch_input : ``float``, optional
        A float number by which to divide the input batches, scaling them. The default is 1.0.

    Methods
    -------
    ``__get_shapes()``
        Method reads and converts shapes of arguments to tuples.
    ``__get_convolved_size()``
        Calculates the shape of convoluted image outputed by a convolutional layer.
    ``forward()``
        Feed foward method from torch.nn.Module.
    ``model_summary()``
        Prints model summary as defined by ``torchsummary``. Method automatically moves model to ``'cpu'``.
    """

    def __init__(self,
                 in_channels: Optional[int] = 3,
                 out_channel: Optional[int] = 16,
                 shape_input: Optional[Union[int, Tuple]] = (84, 84),
                 kernel: Optional[Union[int, Tuple]] = (8, 8),
                 stride: Optional[Union[int, Tuple]] = (4, 4),
                 padding: Optional[Union[int, Tuple]] = (0, 0),
                 out_features_linear: Optional[int] = 256,
                 number_actions: Optional[int] = 4,
                 agent_architecture: Optional[int] = 1,
                 use_batch_norm: Optional[bool] = False,
                 scale_batch_input: Optional[float] = 1.0
                 ) -> None:

        super(DQNModel, self).__init__()
        # PARAMETERS
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.shape_input = self.__get_shapes(shape_input)
        self.kernel = self.__get_shapes(kernel)
        self.stride = self.__get_shapes(stride)
        self.padding = self.__get_shapes(padding)
        self.out_features_linear = out_features_linear
        self.number_actions = number_actions
        self.agent_architecture = agent_architecture
        self.use_batch_norm = use_batch_norm
        self.scale_batch_input = scale_batch_input
        self.conv_out_shape = self.shape_input
        relu_linear = "ReLU_3"

        # SEQUENTIAL CONVOLUTIONAL LAYERS
        # Architecture 1, with two convolutional layers and w/o bach normalization
        self.conv_layers = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("Conv_1", torch.nn.Conv2d(in_channels=self.in_channels,
                                               out_channels=self.out_channel,
                                               kernel_size=self.kernel,
                                               stride=self.stride,
                                               padding=self.padding)),
                    ("ReLU_1", torch.nn.ReLU()),
                    ("Conv_2", torch.nn.Conv2d(in_channels=self.out_channel,
                                               out_channels=int(self.out_channel * 2),
                                               kernel_size=(int(self.kernel[0] / 2),
                                                            int(self.kernel[1] / 2)),
                                               stride=(int(self.stride[0] / 2),
                                                       int(self.stride[1] / 2)),
                                               padding=self.padding)),
                    ("ReLU_2", torch.nn.ReLU())
                ]))
        self.params_conv = [(self.kernel, self.stride, self.padding),
                            ((int(self.kernel[0] / 2), int(self.kernel[1] / 2)),
                             (int(self.stride[0] / 2), int(self.stride[1] / 2)),
                             self.padding)]
        # Architecture 1, with two convolutional layers and w/ bach normalization
        if ((self.agent_architecture == 1) and self.use_batch_norm):
            self.conv_layers = torch.nn.Sequential(
                collections.OrderedDict(
                    [
                        ("Conv_1", torch.nn.Conv2d(in_channels=self.in_channels,
                                                   out_channels=self.out_channel,
                                                   kernel_size=self.kernel,
                                                   stride=self.stride,
                                                   padding=self.padding)),
                        ("Norm_1", torch.nn.BatchNorm2d(self.out_channel)),
                        ("ReLU_1", torch.nn.ReLU()),
                        ("Conv_2", torch.nn.Conv2d(in_channels=self.out_channel,
                                                   out_channels=int(self.out_channel * 2),
                                                   kernel_size=(int(self.kernel[0] / 2),
                                                                int(self.kernel[1] / 2)),
                                                   stride=(int(self.stride[0] / 2),
                                                           int(self.stride[1] / 2)),
                                                   padding=self.padding)),
                        ("Norm_1", torch.nn.BatchNorm2d(int(self.out_channel * 2))),
                        ("ReLU_2", torch.nn.ReLU())
                    ]))
        # Architecture 2, with three convolutional layers w/o batch normalization
        elif ((self.agent_architecture == 2) and (not self.use_batch_norm)):
            self.conv_layers = torch.nn.Sequential(
                collections.OrderedDict(
                    [
                        ("Conv_1", torch.nn.Conv2d(in_channels=self.in_channels,
                                                   out_channels=self.out_channel,
                                                   kernel_size=self.kernel,
                                                   stride=self.stride,
                                                   padding=self.padding)),
                        ("ReLU_1", torch.nn.ReLU()),
                        ("Conv_2", torch.nn.Conv2d(in_channels=self.out_channel,
                                                   out_channels=int(self.out_channel * 2),
                                                   kernel_size=(int(self.kernel[0] / 2),
                                                                int(self.kernel[1] / 2)),
                                                   stride=(int(self.stride[0] / 2),
                                                           int(self.stride[1] / 2)),
                                                   padding=self.padding)),
                        ("ReLU_2", torch.nn.ReLU()),
                        ("Conv_3", torch.nn.Conv2d(in_channels=int(self.out_channel * 2),
                                                   out_channels=int(self.out_channel * 2),
                                                   kernel_size=(int((self.kernel[0] / 2) - 1),
                                                                int((self.kernel[1] / 2) - 1)),
                                                   stride=(int((self.stride[0] / 2) - 1),
                                                           int((self.stride[1] / 2) - 1)),
                                                   padding=self.padding)),
                        ("ReLU_3", torch.nn.ReLU())
                    ]))
            self.params_conv = [(self.kernel, self.stride, self.padding),
                                ((int(self.kernel[0] / 2), int(self.kernel[1] / 2)),
                                 (int(self.stride[0] / 2), int(self.stride[1] / 2)),
                                 self.padding),
                                ((int((self.kernel[0] / 2) - 1), int((self.kernel[1] / 2) - 1)),
                                 (int((self.stride[0] / 2) - 1), int((self.stride[1] / 2) - 1)),
                                 self.padding)]
            relu_linear = "ReLU_4"
        # Architecture 2, with three convolutional layers w/ batch normalization
        if ((self.agent_architecture == 2) and self.use_batch_norm):
            self.conv_layers = torch.nn.Sequential(
                collections.OrderedDict(
                    [
                        ("Conv_1", torch.nn.Conv2d(in_channels=self.in_channels,
                                                   out_channels=self.out_channel,
                                                   kernel_size=self.kernel,
                                                   stride=self.stride,
                                                   padding=self.padding)),
                        ("Norm_1", torch.nn.BatchNorm2d(self.out_channel)),
                        ("ReLU_1", torch.nn.ReLU()),
                        ("Conv_2", torch.nn.Conv2d(in_channels=self.out_channel,
                                                   out_channels=int(self.out_channel * 2),
                                                   kernel_size=(int(self.kernel[0] / 2),
                                                                int(self.kernel[1] / 2)),
                                                   stride=(int(self.stride[0] / 2),
                                                           int(self.stride[1] / 2)),
                                                   padding=self.padding)),
                        ("Norm_2", torch.nn.BatchNorm2d(int(self.out_channel * 2))),
                        ("ReLU_2", torch.nn.ReLU()),
                        ("Conv_3", torch.nn.Conv2d(in_channels=int(self.out_channel * 2),
                                                   out_channels=int(self.out_channel * 2),
                                                   kernel_size=(int((self.kernel[0] / 2) - 1),
                                                                int((self.kernel[1] / 2) - 1)),
                                                   stride=(int((self.stride[0] / 2) - 1),
                                                           int((self.stride[1] / 2) - 1)),
                                                   padding=self.padding)),
                        ("Norm_3", torch.nn.BatchNorm2d(int(self.out_channel * 2))),
                        ("ReLU_3", torch.nn.ReLU())
                    ]))
            self.params_conv = [(self.kernel, self.stride, self.padding),
                                ((int(self.kernel[0] / 2), int(self.kernel[1] / 2)),
                                 (int(self.stride[0] / 2), int(self.stride[1] / 2)),
                                 self.padding),
                                ((int((self.kernel[0] / 2) - 1), int((self.kernel[1] / 2) - 1)),
                                 (int((self.stride[0] / 2) - 1), int((self.stride[1] / 2) - 1)),
                                 self.padding)]
            relu_linear = "ReLU_4"
        # Calculating output shape in the last convolutional layer to know number of input features in first linear layer
        for k, s, p in self.params_conv:
            self.conv_out_shape = self.__get_convolved_size(input_shape=self.conv_out_shape, kernel=k, stride=s, padding=p)
        self.in_features_linear = (int(self.out_channel * 2) * self.conv_out_shape[0] * self.conv_out_shape[1])
        # SEQUENTIAL LINEAR LAYERS
        self.linear_layers = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("Linear_1", torch.nn.Linear(in_features=self.in_features_linear,
                                                 out_features=self.out_features_linear)),
                    (relu_linear, torch.nn.ReLU()),
                    ("Linear_2", torch.nn.Linear(in_features=self.out_features_linear,
                                                 out_features=self.number_actions))
                ]))

    def __get_shapes(self,
                     argument: Any
                     ) -> Tuple:
        r"""
        Helper method to retrieve input frame, kernel, stride and padding shapes.

        Parameters
        ----------
        argument : ``Union[int, tuple, list, numpy.ndarray]``
            Value(s) for the argument shape.

        Raises
        ------
        TypeError
            Argument must be int or tuple, list or numpy.ndarray of size 2.
            Argument must have size 2

        Returns
        -------
        ``tuple``
             A tuple with the (height, width) of argument's shape.
        """
        if (isinstance(argument, int)):
            return (argument, argument)
        elif (isinstance(argument, tuple)):
            if (len(argument) == 2):
                return argument
            else:
                raise TypeError('TypeError! Argument must have size 2. Got size equal to %s...' % str(len(argument)))
        elif (isinstance(argument, list) and (len(argument) == 2)):
            if (len(argument) == 2):
                return tuple(argument)
            else:
                raise TypeError('TypeError! Argument must have size 2. Got size equal to %s...' % str(len(argument)))
        elif (isinstance(argument, np.ndarray) and (len(argument) == 2)):
            if (len(argument) == 2):
                return tuple(argument)
            else:
                raise TypeError('TypeError! Argument must have size 2. Got size equal to %s...' % str(len(argument)))
        else:
            s = str(type(argument))
            raise TypeError('TypeError! Argument must be int or tuple, list or numpy.ndarray of size 2. Got %s...' % s)

    def __get_convolved_size(self,
                             input_shape: Optional[Union[int, Tuple]] = (84, 84),
                             kernel: Optional[Union[int, Tuple]] = (8, 8),
                             stride: Optional[Union[int, Tuple]] = (4, 4),
                             padding: Optional[Union[int, Tuple]] = (0, 0)
                             ) -> Tuple:
        r"""
        Private method to calculate the shape of convoluted image.

        Parameters
        ----------
        input_shape : ``tuple``, optional
            Tuple with image's initial shape (heigh, width). The default is (84, 84).
        kernel : ``tuple``, optional
            Tuple with cnn kernel shape (heigh, width). The default is (8, 8).
        stride : ``tuple``, optional
            Tuple with cnn stride size (vertical, horizontal). The default is (4, 4).
        padding : ``tuple``, optional
            Tuple with cnn padding size (vertical, horizontal). The default is (0, 0).

        Returns
        -------
        ``tuple``
            Convoluted image shape from CNN.

        Notes
        -----
        Shapes calculated based on the following equations:
            Convolved height = (Input H + padding H top + padding H bottom - kernel H) / (stride H) + 1
            Convolved width = (Output W + padding W right + padding W left - kernel W) / (stride W) + 1
        """
        h = ((input_shape[0] + padding[0] + padding[0] - kernel[0]) / (stride[0]) + 1)
        w = ((input_shape[1] + padding[1] + padding[1] - kernel[1]) / (stride[1]) + 1)
        return (int(h), int(w))

    def forward(self,
                input_state: torch.Tensor
                ) -> torch.Tensor:
        r"""
        Method feeds inputs to DQN instantiated.

        Parameters
        ----------
        input_state : ``torch.Tensor``
            Tensor with state observation of shape ``torch.Size([batch, input_channels, frame_height, frame_width])``.

        Returns
        -------
        outputs : ``torch.Tensor``
            Tensor of shape ``torch.Size([batch, number_actions])`` with action values.
        """
        input_state = input_state.type(torch.float) / self.scale_batch_input
        output = self.conv_layers(input_state)
        # output = output.view(-1, (int(self.out_channel * 2) * self.conv_out_shape[0] * self.conv_out_shape[1]))
        output = output.view(output.size(0), -1)
        # output = output.contiguous().view(output.size(0), -1)
        head = self.linear_layers(output)
        return head

    def model_summary(self,
                      device: Optional[str] = 'cpu'
                      ) -> None:
        r"""
        Method returns the table with the models summary generated by ``torchsummary``.

        Parameters
        ----------
        device : ``str``, optional
            Name of device where to move model and parameters, either ``'cpu'`` or ``'cuda'``. The default is ``'cpu'``.

        Returns
        -------
        ``None``
            Prints information.
        """
        mdl_device = torch.device(device)
        return summary(self.to(mdl_device), (self.in_channels, self.shape_input[0], self.shape_input[1]),
                       batch_size=-1, device=device)
