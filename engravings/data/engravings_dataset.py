import os
import cv2
from torch.utils.data import Dataset, DataLoader


class EngravingsDataset(Dataset):
    def __init__(
            self,
            root: str,
            path: str,
            augmentation=None,
    ):
        self.augmentation = augmentation
        self.images, self.labels = self._create_dataset(root,
                                                        path,
                                                        )

    @staticmethod
    def _create_dataset(root, path):
        images = []
        labels = []

        for img_name in os.listdir(os.path.join(root, path)):
            if 'class_0_' in img_name:
                labels.append(0)
            else:
                labels.append(1)

            images.append(os.path.join(root, path, img_name))


        assert len(images) == len(labels)

        return images, labels

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.augmentation:
            img = self.augmentation(image=img)['image']

        return {
            "img": img,
            "label": label,
        }

    def __len__(self):
        return len(self.images)

    def get_dataloader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    from engravings.data import get_train_aug_preproc
    from engravings.models import EngravingsEffnet
    from matplotlib import pyplot as plt


    def visualize(img):
        image = img.copy()
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(image)

    ds = EngravingsDataset(
        '/home/juravlik/PycharmProjects/ssu_artworks/static/data/engravings/gp_wk/',
        'valid',
        get_train_aug_preproc(EngravingsEffnet.get_preprocess_fn()),
    )

    for i in ds:
        if i['label'] == 0:
            continue
        visualize(i['img'])
        print(i['label'])

        plt.waitforbuttonpress()
        plt.close()

