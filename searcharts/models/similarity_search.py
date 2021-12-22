from searcharts.data import get_label_str
from searcharts.models import Embedder
from searcharts.validation import Index
import torch
from searcharts.utils import get_numpy_from_torch
import numpy as np
import pandas as pd


class SimilaritySearch:
    def __init__(self, embedder: Embedder, index: Index, csv_file_with_images_paths: str, label_columns=('class'),
                 labels: dict = None, device=torch.device('cpu')):
        self.labels = labels
        self.label_columns = label_columns
        self.embedder = embedder
        self.index = index
        self.device = device
        self.csv_file = pd.read_csv(csv_file_with_images_paths, sep=';')

    def reset(self):
        self.index.reset()

    def build_index(self, index_loader):
        for data in index_loader:
            images = data['image']
            images = images.to(self.device)
            vectors = self.embedder.forward_batch(images)
            vectors = get_numpy_from_torch(vectors)

            self.index.add_batch(vectors)

        self.index.build_index()

    def search_image(self, image: np.array, n_images: int, return_labels=False):
        """
        if return_classes=True, then func will return labels, else return distances and paths of predicted objects
        """
        embedding = self.embedder.get_inference_for_one_image(image)
        embedding = get_numpy_from_torch(embedding)
        dists, indexes = self.index.predict(embedding, n_images)
        dists, indexes = dists[0], indexes[0]
        if return_labels:
            return self._get_labels_from_indexes(indexes)
        else:
            return dists, self._get_paths_from_indexes(indexes)

    def _get_paths_from_indexes(self, indexes) -> np.array:
        return self.csv_file.iloc[indexes]['imgId'].values

    def search_batch(self, batch_images: torch.Tensor, n_images: int, return_labels=True):
        """
        if return_classes=True, then func will return labels, else return distances and paths of predicted objects
        """
        embeddings_batch = self.embedder.forward_batch(batch_images)
        embeddings_batch = get_numpy_from_torch(embeddings_batch)
        dists, indexes = self.index.predict(embeddings_batch, n_images)
        if return_labels:
            return np.apply_along_axis(self._get_labels_from_indexes, 1, indexes)
        else:
            return dists, np.apply_along_axis(self._get_paths_from_indexes, 1, indexes)

    def _get_labels_from_indexes(self, indexes) -> np.array:
        rows = self.csv_file.iloc[indexes]
        classes = rows.apply(lambda row: get_label_str(row, self.label_columns), axis=1).to_numpy()

        return np.apply_along_axis(lambda class_: self.labels[class_[0]], 0, np.array([classes]))
    