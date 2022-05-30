from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np
import torch
import os


class EngravingsEffnet(nn.Module):
    def __init__(self, efficientnet: str, feature_extracting: bool = True, pretrained=True):
        super(EngravingsEffnet, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(efficientnet)
        else:
            self.model = EfficientNet.from_name(efficientnet)

        self._set_parameter_requires_grad(feature_extracting)

        num_ftrs = self.model._fc.in_features

        self.model._fc = nn.Linear(num_ftrs, 2)

    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model._fc.parameters():
                param.requires_grad = True

    def freeze_only_first_n_layers(self, num_first_layers: int):
        count = 0
        for param in self.model.parameters():
            count += 1
            if count < num_first_layers:
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)

        return x

    @staticmethod
    def get_preprocess_fn():

        def preprocess_input(x, **kwargs):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            if x.max() > 1:
                x = x / 255.0

            mean = np.array(mean)
            x = x - mean

            std = np.array(std)
            x = x / std
            return x

        return preprocess_input

    def get_fc_parameters(self):
        return self.model._fc.parameters()

    def get_parameters(self):
        return self.model.parameters()


if __name__ == '__main__':

    model = EngravingsEffnet(
        efficientnet='efficientnet-b0',
        feature_extracting=True
    )

    # for param in model.get_parameters():
    for i in filter(lambda x: x.requires_grad, model.get_parameters()):
        print(i)

