from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from searcharts.utils import open_image_RGB



def get_label_str(row, label_columns) -> str:
    rez = ''
    for label in label_columns:
        rez += row[label].strip() + ' '
    return rez.strip()


class ArcFaceDataset(Dataset):
    labels = dict()

    def __init__(self,
                 csv_path: str,
                 root_to_data,
                 augmentation,
                 mode='val',
                 length=0,
                 name_of_images='img.jpg',
                 label_columns: list = ('class')
                 ):
        self.augmentation = augmentation
        self.name_of_images = name_of_images
        self.mode = mode
        self.length = length
        self.root_to_data = root_to_data
        self.plates_data = pd.read_csv(csv_path, sep=';')
        self.label_columns = label_columns
        self.get_labels()
        #self.aug_for_original_image = get_valid_augmentation() #_get_validation_augmentation()

    def get_labels(self):
        str_labels = self.plates_data.groupby(self.label_columns, as_index=False).all().apply(
            lambda row: self._get_label_str(row), axis=1
        )
        for pattern in str_labels:
            self.get_label(pattern)

    def _get_label_str(self, row) -> str:
        return get_label_str(row, self.label_columns)

    def get_label(self, pattern):
        if pattern not in self.labels:
            self.labels[pattern] = len(self.labels)
        return self.labels[pattern]

    def __getitem__(self, index: int):
        row = self.plates_data.iloc[index]
        imgId = row['imgId']
        img_path = os.path.join(
            os.path.join(self.root_to_data, imgId),
            self.name_of_images
        )
        original_image = open_image_RGB(img_path)
        img = self.augmentation(image=original_image)['image']

        pattern = self._get_label_str(row)
        label = self.get_label(pattern)

        return {
            "image": img,
            "label": label,
            "imgId": imgId
        }

    def __len__(self):
        if self.mode == 'train' and self.length != 0:
            return self.length
        return self.plates_data.shape[0]

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    from searcharts.data import get_valid_aug_preproc, get_train_aug_preproc
    from searcharts.models import ArtEfficientnet

    ds = ArcFaceDataset(
        '/home/juravlik/PycharmProjects/ssu_artworks/static/data/csv/train.csv',
        '/home/juravlik/PycharmProjects/ssu_artworks/static/data/data',
        get_train_aug_preproc(ArtEfficientnet.get_preprocess_fn()),
        label_columns=['class'])
    for i in ds:
        print(i['label'], i['imgId'])

        break

