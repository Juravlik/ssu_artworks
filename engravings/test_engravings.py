import os
import pandas as pd
import torch
import numpy as np
from PIL import Image


from searcharts.utils import get_param_from_config, object_from_dict
from engravings.data import get_valid_aug_preproc
from engravings.models import EngravingsEffnet

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

basedir = os.path.abspath(os.path.dirname(__file__))

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
CONFIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../configs"))

config_path = os.path.join(CONFIG_DIR, 'test_engravings_config.yaml')
config = get_param_from_config(config_path)


device = torch.device(config.device)
model_checkpoint_path = config.model_checkpoint_path
DATA_PATH = config.DATA_PATH

model_checkpoint = torch.load(model_checkpoint_path, map_location=device)
model = object_from_dict(model_checkpoint['config']['model'])
model.load_state_dict(model_checkpoint['model'])
model.to(device)
model.eval()

preprocessing_for_model = get_valid_aug_preproc(model.get_preprocess_fn())


def get_model_answer(target_path: str):

    target_image = Image.open(target_path).convert("RGB")
    target_image = np.array(target_image)

    image = preprocessing_for_model(image=target_image)['image']
    image = torch.from_numpy(image)[None, :, :, :].to(device)

    scores = model(image)

    scores = torch.softmax(torch.flatten(scores), dim=0)

    return torch.argmax(torch.flatten(scores), dim=0).detach().cpu().numpy()


if __name__ == "__main__":

    image_paths = os.listdir(DATA_PATH)

    labels = []
    predictions = []

    for img_path in image_paths:
        prediction = get_model_answer(os.path.join(DATA_PATH, img_path))
        predictions.append(prediction)
        labels.append(int('class_1_' in img_path))

    print(confusion_matrix(labels, predictions))

    print('Accuracy: ', accuracy_score(labels, predictions))
    print('Recall: ', recall_score(labels, predictions))
    print('Precision: ', precision_score(labels, predictions))
