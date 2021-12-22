from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np
import torch


class ArtEfficientnet(nn.Module):
    def __init__(self, efficientnet: str, vector_size: int, feature_extracting: bool = True, pretrained=True):
        super(ArtEfficientnet, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(efficientnet)
        else:
            self.model = EfficientNet.from_name(efficientnet)

        output_size = self.model._fc.out_features

        self._set_parameter_requires_grad(feature_extracting)

        self.embeddeer = nn.Sequential(*[
            nn.Linear(output_size, vector_size),
        ])
        self.activate = nn.ReLU()

    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model._fc.parameters():
                param.requires_grad = True

            for param in [*self.model._conv_head.parameters()]+[*self.model._bn1.parameters()]:
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.activate(x)
        x = self.embeddeer(x)

        return x

    @staticmethod
    def get_preprocess_fn():

        def preprocess_input(x, **kwargs):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            input_range = [0, 1]

            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0

            mean = np.array(mean)
            x = x - mean

            std = np.array(std)
            x = x / std
            return x

        return preprocess_input

    def get_fc_parameters(self):
        return [*self.model._fc.parameters()] + [*self.embeddeer.parameters()]

    def get_cnn_parameters(self):
        return [*self.model._conv_stem.parameters()] + [*self.model._bn0.parameters()] + [
            *self.model._blocks.parameters()] + [*self.model._conv_head.parameters()] + [*self.model._bn1.parameters()]


if __name__ == '__main__':
    net = ArtEfficientnet('efficientnet-b0', 128, False)

    # print(np.asarray([i.shape for i in [*filter(lambda x: x.requires_grad, net.parameters())]]))
    # print()
    # print()
    print(np.asarray([i.shape for i in [*filter(lambda x: x.requires_grad, net.get_cnn_parameters())]]))
    print(np.asarray([i.shape for i in [*filter(lambda x: x.requires_grad, net.get_fc_parameters())]]))
    # print(sum([(i.shape==j.shape) for i,j in zip(net.parameters(),net.get_cnn_parameters()+net.get_fc_parameters())]))
    # print(len([*net.parameters()]),len([*net.get_cnn_parameters()+net.get_fc_parameters()]))