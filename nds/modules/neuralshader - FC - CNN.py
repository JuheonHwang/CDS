
from nds.modules.fc import FC
from nds.modules.cnn import CONV
from nds.modules.gfft import GaussianFourierFeatureTransform
from nds.modules.embedder import get_embedder

import numpy as np
import torch

class NeuralShader(torch.nn.Module):

    def __init__(self,
                 hidden_features_size=256,
                 hidden_features_layers=7,
                 kernel_size=11,
                 stride=1,
                 activation='relu',
                 last_activation=None,
                 fourier_features='none',
                 mapping_size=256,
                 fft_scale=10,
                 device='cpu'):

        super().__init__()
        self.fourier_feature_transform = None
        if fourier_features == 'gfft':
            self.fourier_feature_transform = GaussianFourierFeatureTransform(3, mapping_size=mapping_size//2, scale=fft_scale, device=device)
            self.diffuse = FC(mapping_size, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None).to(device)
            self.specular = CONV(hidden_features_size+3+3, 3, kernel_size, stride, [hidden_features_size // 2], activation, last_activation).to(device)
        elif fourier_features == 'positional':
            self.fourier_feature_transform, channels = get_embedder(fft_scale)
            self.diffuse = FC(channels, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None).to(device)
            self.specular = CONV(hidden_features_size+3+3, 3, kernel_size, stride, [hidden_features_size // 2], activation, last_activation).to(device)
        elif fourier_features == 'none':
            self.diffuse = FC(3, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None).to(device)
            self.specular = CONV(hidden_features_size+3+3, 3, kernel_size, stride, [hidden_features_size // 2], activation, last_activation).to(device)

        # Store the config
        self._config = {
            'hidden_features_size': hidden_features_size,
            'hidden_features_layers': hidden_features_layers,
            'kernel_size': kernel_size,
            'stride': stride,
            'activation': activation,
            'last_activation': last_activation,
            'fourier_features': fourier_features,
            'mapping_size': mapping_size,
            'fft_scale': fft_scale
        }

    def forward(self, position, normal, view_dir):
        #! The current apporach only transforms the positions
        #! However it is also possible to transform all the input (views and normals)
        height, width, _ = position.shape
        if self.fourier_feature_transform is not None:
            h = self.diffuse(self.fourier_feature_transform(position.view(-1, 3)))
            h = torch.cat([h.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0), normal.permute(2, 0, 1).unsqueeze(0), view_dir.permute(2, 0, 1).unsqueeze(0)], dim=1)
        else:
            h = self.diffuse(position)
            h = torch.cat([h.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0), normal.permute(2, 0, 1).unsqueeze(0), view_dir.permute(2, 0, 1).unsqueeze(0)], dim=1)

        return self.specular(h).squeeze().permute(1, 2, 0)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)

        # Convert data between versions
        version = data['version']

        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'])

        if version < 2 and isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            print("Warning: B matrix for GFFT features is not stored in checkpoints of versions < 2")
        elif isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            shader.fourier_feature_transform.B = data['B']
        return shader

    def save(self, path):
        data = {
            'version': 2,
            'config': self._config,
            'state_dict': self.state_dict()
        }

        if isinstance(self.fourier_feature_transform, GaussianFourierFeatureTransform):
            data['B'] = self.fourier_feature_transform.B
        torch.save(data, path)