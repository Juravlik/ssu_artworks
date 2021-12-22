import cv2
import os


def open_image_RGB(path_to_open):
    image = cv2.imread(path_to_open)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image_RGB(image, path_to_save):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_to_save, image)


def get_image_name_that_not_exists(upload_path, file_name):
    i = 0
    part_path = '.'.join(file_name.split('.')[:-1])
    while os.path.exists(os.path.join(upload_path, part_path + str(i) + '.jpg')):
        i += 1
    else:
        file_name = part_path + str(i) + '.jpg'

    return file_name


def save_mask_2d(mask, path_to_save):
    cv2.imwrite(path_to_save, mask)