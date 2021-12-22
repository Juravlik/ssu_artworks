from searcharts.utils.factory import object_from_dict
import numpy as np
import torch


class Embedder:
    def __init__(self, embedder, preprocessing=None, device=torch.device('cpu')):
        self.embedder = embedder
        self.preprocessing = preprocessing
        self.device = device
        self.embedder.eval()

    def forward_batch(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embedder(images)

    def get_inference_for_one_image(self, image: np.array) -> torch.Tensor:
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
        image = torch.from_numpy(image)[None, :, :, :].to(self.device)
        embedding = self.forward_batch(image)
        return embedding

    @staticmethod
    def get_model_with_loaded_weights(config, weights, param):
        return object_from_dict(config, param).load_state_dict(weights)
