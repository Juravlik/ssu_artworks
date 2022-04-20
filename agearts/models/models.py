from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np
import torch
import os


class AgeArtsEffnet(nn.Module):
    def __init__(self, efficientnet: str, feature_extracting: bool = True, pretrained=True):
        super(AgeArtsEffnet, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(efficientnet)
        else:
            self.model = EfficientNet.from_name(efficientnet)

        self._set_parameter_requires_grad(feature_extracting)

        num_ftrs = self.model._fc.in_features

        self.model._fc = nn.Linear(num_ftrs, 1)

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


class AgeArtEfficientnet(nn.Module):
    def __init__(self, path_to_searchart_model: str, feature_extracting: bool = True):
        super(AgeArtEfficientnet, self).__init__()

        model_checkpoint = torch.load(os.path.join(path_to_searchart_model),
                                      map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.model = ArtEfficientnet(
            efficientnet=model_checkpoint['config']['model']['efficientnet'],
            vector_size=model_checkpoint['config']['embedding_size'],
            feature_extracting=feature_extracting
        )

        self.model.load_state_dict(model_checkpoint['model'])

        self._set_parameter_requires_grad(feature_extracting)

        self.new_fc = nn.Sequential(*[
            nn.Linear(model_checkpoint['config']['embedding_size'], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ])

        #nn.Linear(model_checkpoint['config']['embedding_size'], 1)

    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

            # for param in self.model._fc.parameters():
            #     param.requires_grad = True

    def freeze_only_first_n_layers(self, num_first_layers: int):
        count = 0
        for param in self.model.parameters():
            count += 1
            if count < num_first_layers:
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = self.new_fc(x)

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
        return self.new_fc.parameters()

    def get_parameters(self):
        return [*self.model.parameters()] + [*self.new_fc.parameters()]


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


    model = AgeArtEfficientnet(
        path_to_searchart_model='/home/juravlik/PycharmProjects/ssu_artworks/models/embedder_b0__fe_false__lr001__s25__m03__emb128/checkpoint.pt',
        feature_extracting=False
    )

    # for param in model.get_parameters():
    for i in filter(lambda x: x.requires_grad, model.get_parameters()):
        print(i)

