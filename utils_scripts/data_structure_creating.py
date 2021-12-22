import pandas as pd
import os
from shutil import copy


PATH_SOURCE_RAW_DATA = '/home/juravlik/Downloads/wikiart'
PATH_DIST = '/home/juravlik/PycharmProjects/ssu_artworks/static/data/data/'
PATH_CSV = '/home/juravlik/PycharmProjects/ssu_artworks/static/data/csv/'

if not os.path.exists(PATH_DIST):
    os.makedirs(os.path.dirname(PATH_DIST), exist_ok=True)
    print('Folder {} was created'.format(PATH_DIST))

if not os.path.exists(PATH_CSV):
    os.makedirs(os.path.dirname(PATH_CSV), exist_ok=True)
    print('Folder {} was created'.format(PATH_CSV))


dirs_list = os.listdir(PATH_SOURCE_RAW_DATA)


for dir in dirs_list:

    print(dir)

    files_list = []

    for file in os.listdir(os.path.join(PATH_SOURCE_RAW_DATA, dir)):
        if file.endswith('.jpg'):
            files_list.append(file.split('.jpg')[0])

            source_path = os.path.join(PATH_SOURCE_RAW_DATA, dir, file)
            dest_path = os.path.join(PATH_DIST, dir + '__' + file.split('.jpg')[0], 'img.jpg')

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            copy(source_path, dest_path)

    df = pd.DataFrame({'filename': files_list})
    df['style'] = dir

    dest_csv = os.path.join(PATH_CSV, dir + '.csv')
    df.to_csv(dest_csv,
              index=False, sep=';')




