import numpy as np
import torch
import torch.nn as nn
from typing import List


class Sine(nn.Module):
    r"""Applies the sine function with frequency scaling element-wise:

    :math:`\text{Sine}(x)= \sin(\omega * x)`

    Args:
        omega: factor used for scaling the frequency

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

def make_module(module):
    # Create a module instance if we don't already have one
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()
        
class Conv2dBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, bias=True, activation=torch.nn.ReLU):
        super().__init__()

        self.conv2d = torch.nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, bias=bias, padding='same')
        #self.norm2d = torch.nn.BatchNorm2d(num_features=dim_out)
        self.activation = make_module(activation) if activation is not None else torch.nn.Identity()

    def forward(self, input):
        return self.activation(self.conv2d(input))
        #return self.activation(self.norm2d(self.conv2d(input)))
        
class DownBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, input):
        return self.pool2d(input)
        
class UpBlock(torch.nn.Module):
    def __init__(self, dim_in, din_out):
        super().__init__()
        self.upconv2d = torch.nn.ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, kernel_size=2, stride=2)
        
    def forward(self, input):
        return self.upconv2d(input)


def siren_init_first(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-1 / n, 
                                     1 / n)

def siren_init(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    omega = kwargs['omega']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-np.sqrt(6 / n) / omega, 
                                     np.sqrt(6 / n) / omega)

def init_weights_normal(**kwargs):
    module = kwargs['module']
    #if isinstance(module, nn.Linear):
    #    if hasattr(module, 'weight'):
    #        nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
    #    if hasattr(module, 'bias'):
    #        nn.init.zeros_(module.bias)
    if hasattr(module, 'weight'):
        nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
    if hasattr(module, 'bias'):
        nn.init.zeros_(module.bias)

def init_weights_normal_last(**kwargs):
    module = kwargs['module']
    #if isinstance(module, nn.Linear):
    #    if hasattr(module, 'weight'):
    #        nn.init.xavier_normal_(module.weight, gain=1)
    #        module.weight.data = -torch.abs(module.weight.data)
    #    if hasattr(module, 'bias'):
    #        nn.init.zeros_(module.bias)
    if hasattr(module, 'weight'):
        nn.init.xavier_normal_(module.weight, gain=1)
        module.weight.data = -torch.abs(module.weight.data)
    if hasattr(module, 'bias'):
        nn.init.zeros_(module.bias)
            
        
class UNET_diffuse(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, hidden_features: List[int], activation='relu', last_activation=None, bias=True, first_omega=30, hidden_omega=30.0):
        super().__init__()

        layers = []

        activations_and_inits = {
            'sine': (Sine(first_omega),
                     siren_init,
                     siren_init_first,
                     None),
            'relu': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     None),
            #'relu': (nn.ReLU(inplace=True),
            #         init_weights_normal_last,
            #         init_weights_normal_last,
            #         None),
            'relu2': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     init_weights_normal_last),
            #'softplus': (nn.Softplus(),
            #            init_weights_normal,
            #            None)
            'softplus': (nn.Softplus(),
                        init_weights_normal_last,
                        init_weights_normal_last,
                        None)
        }

        activation_fn, weight_init, first_layer_init, last_layer_init = activations_and_inits[activation]


        # First layer
        self.conv1 = nn.Sequential(
            self.make_Conv2d(in_features, hidden_features[0], kernel_size, stride, bias=bias, activation=activation_fn, layer_init=first_layer_init)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second layer
        self.conv2 = nn.Sequential(
            self.make_Conv2d(hidden_features[0], hidden_features[1], kernel_size, stride, bias=bias, activation=activation_fn, layer_init=weight_init)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third layer
        self.bottleneck = nn.Sequential(
            self.make_Conv2d(hidden_features[1], hidden_features[2], kernel_size, stride, bias=bias, activation=activation_fn, layer_init=weight_init)
        )
        
        # Forth layer
        self.upconv1 = nn.ConvTranspose2d(in_channels=hidden_features[2], out_channels=hidden_features[1], kernel_size=2, stride=2)
        self.upconv1.apply(lambda module: weight_init(module=module, n=hidden_features[2]))
        
        self.conv4 = nn.Sequential(
            self.make_Conv2d(hidden_features[2], hidden_features[1], kernel_size, stride, bias=bias, activation=activation_fn, layer_init=weight_init)
        )
        
        # Fifth layer
        self.upconv2 = nn.ConvTranspose2d(in_channels=hidden_features[1], out_channels=hidden_features[0], kernel_size=2, stride=2)
        self.upconv1.apply(lambda module: weight_init(module=module, n=hidden_features[1]))
        
        self.conv5 = nn.Sequential(
            self.make_Conv2d(hidden_features[1], hidden_features[0], kernel_size, stride, bias=bias, activation=activation_fn, layer_init=weight_init)
        )
        
        # Final layer
        self.conv6 = nn.Sequential(
            self.make_Conv2d(hidden_features[0], out_features, kernel_size, stride, bias=bias, activation=last_activation, layer_init=weight_init)
        )
        
    def make_Conv2d(self, dim_in, dim_out, kernel_size, stride=1, bias=True, activation=torch.nn.ReLU, layer_init=None):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, bias=bias, padding='same'),
            # nn.BatchNorm2d(num_features=dim_out),
            make_module(activation) if activation is not None else nn.Identity()
        )
        if layer_init is not None:
            layer.apply(lambda module: layer_init(module=module, n=dim_in))
        return layer

    def forward(self, x):
        layer1 = self.conv1(x)
        out = self.pool1(layer1)
        
        layer2 = self.conv2(out)
        out = self.pool2(layer2)
        
        bottleneck = self.bottleneck(out)
        
        upconv1 = self.upconv1(bottleneck)
        cat1 = torch.cat([layer2, upconv1], dim=1)
        layer4 = self.conv4(cat1)
        
        upconv2 = self.upconv2(layer4)
        cat2 = torch.cat([layer1, upconv2], dim=1)
        layer5 = self.conv5(cat2)
        
        layer6 = self.conv6(layer5)
        
        return layer6