import os
import pandas as pd
import torch
import numpy as np
from PIL import Image


from searcharts.utils import get_param_from_config, object_from_dict
from searcharts.data import get_valid_aug_preproc
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from searcharts.models import Embedder, SimilaritySearch

from sklearn.metrics import mean_squared_error, mean_absolute_error


basedir = os.path.abspath(os.path.dirname(__file__))

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
CONFIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../configs"))

config_path = os.path.join(CONFIG_DIR, 'test_age_config.yaml')
config = get_param_from_config(config_path)


n_images = config.n_images
device = torch.device(config.device)
embedder_checkpoint_path = config.embedder_checkpoint_path
UPLOADED_PATH = os.path.join(basedir, config.UPLOADED_PATH)
CSV_PATH = config.CSV_PATH
TEST_PATH = config.TEST_PATH
DATA_PATH = config.DATA_PATH

embedder_checkpoint = torch.load(embedder_checkpoint_path, map_location=device)
embedding_size = embedder_checkpoint['config']['embedding_size']
embedder = object_from_dict(embedder_checkpoint['config']['model'], vector_size=embedding_size)
embedder.load_state_dict(embedder_checkpoint['model'])
embedder.to(device)
embedder.eval()
preprocessing_for_embedder = get_valid_aug_preproc(embedder.get_preprocess_fn())
class_embedder = Embedder(embedder, preprocessing_for_embedder, device)

df = pd.read_csv(CSV_PATH, sep=';')

indexer = object_from_dict(
    config.index,
    device=device,
    dimension=embedding_size,
)

indexer.load_ranking_model(path_to_load_model=config.ranking_model_path)

similarity_search = SimilaritySearch(embedder=class_embedder, index=indexer, device=device,
                                     csv_file_with_images_paths=CSV_PATH)


def search_similar(target_path: str):

    image = Image.open(target_path).convert("RGB")
    target_image = np.array(image)

    dists, similar_paths = similarity_search.search_image(target_image,
                                                          n_images=n_images,
                                                          return_labels=False)
    similar_paths = similar_paths.tolist()

    data = paths_to_data(similar_paths)

    age = define_age(data)

    return target_image, similar_paths, age


def paths_to_data(paths):
    global df

    cur_df = df[df['imgId'].isin(paths)]

    all_data = cur_df[['imgId', 'style', 'author', 'age']].values

    all_data_sort = []

    for path in paths:
        for data in all_data:
            if path in data:
                all_data_sort.append(data.tolist())

    return all_data_sort


def define_age(data):

    data = [data[i][3] for i in range(len(data))]

    res = np.nanmean(data)

    if res > 0:
        return int(res)
    else:
        return 1800

if __name__ == "__main__":
    df_test = pd.read_csv(TEST_PATH, sep=';')

    imgIds = df_test['imgId'].values
    ages = df_test['age'].values

    preds_age = []

    for i in range(len(ages)):
        _, _, pred_age = search_similar(os.path.join(DATA_PATH, imgIds[i], 'img.jpg'))
        preds_age.append(pred_age)

        print(ages[i], pred_age)

    print('RMSE: ', mean_squared_error(ages, preds_age, squared=False))
    print('MAE: ', mean_absolute_error(ages, preds_age))
