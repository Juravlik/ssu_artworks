import numpy as np


class Metric:
    def __init__(self, k: int):
        self.k = k
        self.result = 0
        self.num_queries = 0

    def update(self, predicts, queries):
        pass

    def get_result(self):
        pass

    def reset(self):
        self.result = 0
        self.num_queries = 0


class OneRecallAtK(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, predicts: np.array, queries: np.array):
        self.result += (predicts[:, :self.k] == queries.reshape(-1, 1)).any(axis=-1).sum()
        self.num_queries += len(queries)

    def get_result(self):
        return self.result / self.num_queries


if __name__ == '__main__':
    predict_1 = np.array([[1, 1, 1], [5, 6, 2], [5, 3, 3]])
    query_1 = np.array([1, 2, 3])
    metric = OneRecallAtK(k=2)
    metric.update(predict_1, query_1)
    metric.reset()
    predict_2 = np.array([[66, 11, 11], [18, 16, 10], [19, 20, 1]])
    query_2 = np.array([10, 16, 10])
    metric.update(predict_2, query_2)
    print(metric.get_result())
