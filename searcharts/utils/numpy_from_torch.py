import torch
import numpy as np


def get_numpy_from_torch(embedding: torch.Tensor) -> np.array:
    return embedding.cpu().data.numpy()