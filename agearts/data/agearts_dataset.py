from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from searcharts.utils import open_image_RGB


class AgeArtsDataset(Dataset):
    labels = dict()

    def __init__(self,
                 csv_path: str,
                 root_to_data,
                 augmentation,
                 mode='val',
                 length=0,
                 name_of_images='img.jpg',
                 label_columns: list = ('age')
                 ):
        self.augmentation = augmentation
        self.name_of_images = name_of_images
        self.mode = mode
        self.length = length
        self.root_to_data = root_to_data
        self.plates_data = pd.read_csv(csv_path, sep=';')
        self.label_columns = label_columns

    def __getitem__(self, index: int):
        row = self.plates_data.iloc[index]
        imgId = row['imgId']
        age = row['age']

        img_path = os.path.join(
            os.path.join(self.root_to_data, imgId),
            self.name_of_images
        )

        original_image = open_image_RGB(img_path)
        img = self.augmentation(image=original_image)['image']

        return {
            "img": img,
            "age": (age - 1401) / (2012 - 1401)
        }

    def __len__(self):
        if self.mode == 'train' and self.length != 0:
            return self.length
        return self.plates_data.shape[0]

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    from agearts.data import get_valid_aug_preproc, get_train_aug_preproc
    from agearts.models import AgeArtsEffnet
    import cv2
    from matplotlib import pyplot as plt


    def visualize(img):
        image = img.copy()
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(image)


    #################
    ds = AgeArtsDataset(
        '/home/juravlik/PycharmProjects/ssu_artworks/static/data/csv/age_train.csv',
        '/home/juravlik/PycharmProjects/ssu_artworks/static/data/data',
        get_train_aug_preproc(AgeArtsEffnet.get_preprocess_fn()),
        label_columns=['age'])

    for i in ds:

        visualize(i['img'])
        plt.waitforbuttonpress()
        plt.close()

