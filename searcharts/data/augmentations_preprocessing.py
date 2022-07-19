import albumentations as A
import torch
import random
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Callable, Optional


IMAGE_SIZE = 224 # 300


def lock_deterministic(seed: int =42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_padding_to_square(x: np.array, **kwargs):

    max_side = max(x.shape)

    return A.PadIfNeeded(
        min_height=max_side, min_width=max_side, always_apply=True, border_mode=cv2.BORDER_CONSTANT
    )(image=x)['image']


def _get_validation_augmentation():
    transforms = [
        A.Lambda(image=add_padding_to_square, mask=add_padding_to_square, always_apply=True),
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
    ]

    return A.Compose(transforms)


def _get_training_augmentation():
    transforms = [

        A.Blur(blur_limit=(3, 3), p=0.1),

        A.OneOf(
            [
                A.ISONoise(color_shift=(0.05, 0.01), intensity=(0.1, 0.5), p=0.1),
                A.IAAAdditiveGaussianNoise(p=0.1),
                A.IAAPerspective(p=0.1),
            ], p=0.3
        ),

        A.RandomBrightnessContrast(p=0.1),

        A.RandomRotate90(p=0.2),

        A.Flip(p=0.2),

        _get_validation_augmentation(),

        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.2),

    ]

    return A.Compose(transforms)


def get_train_aug_preproc(preprocessing_fn: Callable):

    return A.Compose([*_get_training_augmentation()] + [*_get_preprocessing(preprocessing_fn)])


def get_valid_aug_preproc(preprocessing_fn):
    return A.Compose([*_get_validation_augmentation()] + [*_get_preprocessing(preprocessing_fn)])


def to_tensor(x: np.array, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def _get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)


if __name__ == "__main__":
    from searcharts.models import ArtEfficientnet

    print(
        [*_get_training_augmentation()] + [*_get_preprocessing(ArtEfficientnet.get_preprocess_fn())]
    )