import torch
import numpy as np
from searcharts.models import SimilaritySearch
from searcharts.validation import Metric
from searcharts.utils import get_numpy_from_torch


class Evaluator:
    def __init__(self, similarity_search: SimilaritySearch, metrics: [Metric], device=torch.device('cpu')):
        self.similarity_search = similarity_search
        self.device = device
        self.metrics = metrics
        self.max_num_queries = max([metric.k for metric in self.metrics])

    def update_all_metrics(self, predict_labels, labels):
        for metric in self.metrics:
            metric.update(predict_labels, labels)

    def reset(self):
        self.reset_all_metrics()
        self.similarity_search.reset()

    def build_index(self, index_loader):
        self.similarity_search.build_index(index_loader)

    def reset_all_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def compute_metrics(self, search_loader) -> np.array:
        for data in search_loader:
            images = data['image']
            images = images.to(self.device)
            labels = get_numpy_from_torch(data['label'])
            predict_labels = self.get_predict_labels(batch_images=images, n_images=self.max_num_queries)
            self.update_all_metrics(predict_labels, labels)

        result = np.asarray([metric.get_result() for metric in self.metrics])
        return result

    def get_predict_labels(self, batch_images: torch.Tensor, n_images: int) -> np.array:
        return self.similarity_search.search_batch(batch_images=batch_images, n_images=n_images, return_labels=True)