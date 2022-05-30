import os
from base64 import b64encode
from datetime import timedelta
from typing import Tuple, List

import pandas as pd
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
from engravings.data import get_valid_aug_preproc
from engravings.models import EngravingsEffnet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)
app.config['SESSION_FILE_THRESHOLD'] = 20

app.secret_key = os.urandom(10)
Session().init_app(app)

basedir = os.path.abspath(os.path.dirname(__file__))

config_path = os.path.join(basedir, '../configs/application_engravings.yaml')
config = get_param_from_config(config_path)

device = torch.device(config.device)
model_checkpoint_path = config.model_checkpoint_path
UPLOADED_PATH = os.path.join(basedir, config.UPLOADED_PATH)

print("Loading model")
model_checkpoint = torch.load(model_checkpoint_path, map_location=device)
model = object_from_dict(model_checkpoint['config']['model'])
model.load_state_dict(model_checkpoint['model'])
model.to(device)
model.eval()
preprocessing_for_model = get_valid_aug_preproc(model.get_preprocess_fn())


def get_model_answer():
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

    image = preprocessing_for_model(image=target_image)['image']
    image = torch.from_numpy(image)[None, :, :, :].to(device)

    scores = model(image)

    scores = torch.softmax(torch.flatten(scores), dim=0)

    scores = scores.detach().cpu().numpy()

    file_object = io.BytesIO()
    img = Image.fromarray(target_image.astype('uint8'))
    img.save(file_object, 'PNG')
    target_image = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')

    return target_image, scores


@app.route('/process', methods=['POST'])
def process():
    if request.files or request.form:
        if "image" in request.files and request.files["image"].filename == "" and request.form["url"] == "":
            return jsonify({})
        target_image, scores = get_model_answer()

        return jsonify({"target_image": target_image, "scores": scores})
    else:
        return jsonify({})


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.files or request.form:
            if request.files["image"].filename == "" and request.form["url"] == "":
                return redirect('/')

            target_image, scores = get_model_answer()

            session['target_image'] = target_image
            session['scores'] = scores

            return redirect('/')
    else:
        target_image = session.get('target_image')
        scores = session.get('scores')

        if target_image and scores is not None:

            delete_sessions()

            text1, text2 = text_by_scores(scores)

            return render_template('index.html', target_image=target_image, text1=text1, text2=text2)
        else:
            return render_template('index.html')


def delete_sessions():
    session.pop('scores', None)
    session.pop('target_image', None)


def text_by_scores(scores):
    argmax = np.argmax(scores)
    score = round(scores[argmax] * 100, 2)

    if argmax == 0:
        text1 = "This is not an engraving by Dürer"

    else:
        text1 = "This is an engraving by Dürer"

    text2 = "with a probability of {}%".format(score)

    return text1, text2


print("Service-started")

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')
