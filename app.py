import os
from base64 import b64encode
from datetime import timedelta
from typing import Dict, List, Tuple, Union, Callable, Optional
import pandas as pd
from searcharts.utils.image_utils import open_image_RGB
from flask import Flask, request, redirect, render_template, session, jsonify
from flask_session import Session
import csv
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import io
import sys


from searcharts.utils import get_param_from_config, object_from_dict
from searcharts.data import get_valid_aug_preproc
from searcharts.models import Embedder, SimilaritySearch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)
app.config['SESSION_FILE_THRESHOLD'] = 20

app.secret_key = os.urandom(10)
Session().init_app(app)

basedir = os.path.abspath(os.path.dirname(__file__))

config_path = os.path.join(basedir, 'configs/application_config.yaml')
config = get_param_from_config(config_path)

n_images = config.n_images
device = torch.device(config.device)
embedder_checkpoint_path = config.embedder_checkpoint_path
UPLOADED_PATH = os.path.join(basedir, config.UPLOADED_PATH)
CSV_PATH = config.CSV_PATH
DATA_PATH = config.DATA_PATH

print("Loading embedder")
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

print("Loading ranker")
indexer.load_ranking_model(path_to_load_model=config.ranking_model_path)

similarity_search = SimilaritySearch(embedder=class_embedder, index=indexer, device=device,
                                     csv_file_with_images_paths=CSV_PATH)


def search_similar() -> Tuple[str, List[str]]:
    """
    retrieve image file or url parameter from request information and search for similar images.
    Returns:
        Tuple of original image base64 encoded and list of paths to similar images, ordered by similarity.
    """
    if "image" in request.files and request.files["image"].filename != "":
        np_file = np.fromfile(request.files["image"], np.uint8)
        target_image = cv2.imdecode(np_file, cv2.IMREAD_COLOR)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    else:
        url = request.form["url"]
        r = requests.get(url, allow_redirects=True)
        stream = BytesIO(r.content)
        image = Image.open(stream).convert("RGB")
        stream.close()
        target_image = np.array(image)


    dists, similar_paths = similarity_search.search_image(target_image,
                                                          n_images=n_images,
                                                          return_labels=False)

    similar_paths = similar_paths.tolist()

    file_object = io.BytesIO()
    img = Image.fromarray(target_image.astype('uint8'))
    img.save(file_object, 'PNG')
    target_image = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return target_image, similar_paths


@app.route('/process', methods=['POST'])
def process():
    if request.files or request.form:
        if "image" in request.files and request.files["image"].filename == "" and request.form["url"] == "":
            return jsonify({})
        target_image, similar_paths = search_similar()
        data = paths_to_data(similar_paths)

        return jsonify({"target_image": target_image, "similar_paths": data})
    else:
        return jsonify({})


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.files or request.form:
            if request.files["image"].filename == "" and request.form["url"] == "":
                return redirect('/')

            target_image, similar_paths = search_similar()

            session['paths'] = similar_paths
            session['target_image'] = target_image

            return redirect('/')
    else:
        paths = session.get('paths')
        target_image = session.get('target_image')

        if paths and target_image:
            data = paths_to_data(paths)

            delete_sessions()

            return render_template('index.html', target_image=target_image, data=data, age=define_age(data))
        else:
            return render_template('index.html')


@app.route('/api/getApiToken/', methods=['POST'])
def response_text():
    content = request.json
    print(content)
    return jsonify({"token": 'LOL'})


@app.route('/api/uploadFile/', methods=['POST'])
def response_images():
    path_target_image = request.json['path_target_image']

    target_image = open_image_RGB(path_target_image)

    dists, similar_paths = similarity_search.search_image(target_image,
                                                          n_images=n_images,
                                                          return_labels=False)

    data = paths_to_data(similar_paths)

    return jsonify({"similar_paths": data, "age": define_age(data)})



def delete_sessions():
    session.pop('paths', None)
    session.pop('target_image', None)


def paths_to_data(paths: List[str]) -> List[List[str]]:
    global df

    cur_df = df[df['imgId'].isin(paths)]

    all_data = cur_df[['imgId', 'style', 'author', 'age']].values

    all_data_sort = []

    for path in paths:
        for data in all_data:
            if path in data:
                all_data_sort.append(data.tolist())

    return all_data_sort


def define_age(data: List[str]) -> int:

    data = [data[i][3] for i in range(len(data))]

    res = np.nanmean(data)

    if res > 0:
        return int(res)
    else:
        return 1800


print("Service-started")

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')
