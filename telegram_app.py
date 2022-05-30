import io
import json
import math
import os
import shutil
from datetime import datetime

import numpy as np
import requests
import telebot
from PIL import Image
from telebot import types

from configs.tg_configs.token import TOKEN
import configs.tg_configs.creditials as cfg

bot = telebot.TeleBot(TOKEN)
user_state = {}

start_reply = types.ReplyKeyboardMarkup(True)
start_reply.row('/start')

start_processing_reply = types.ReplyKeyboardMarkup(True)
start_processing_reply.row('/find_similar')


def user_exist(user_id):
    return user_id in user_state


@bot.message_handler(commands=['start'])
def start_message(message):
    user_id = message.from_user.id

    if user_exist(user_id):
        user_state[user_id]['image'] = None

    else:
        user_state[user_id] = {'image': None,
                               'n_images': 10}

    bot.send_message(
        message.chat.id,
        'Hello! I am a ssu_artworks_bot. I will try to find similar artworks for you!',
    )

    bot.send_message(
        message.chat.id,
        'Just upload an image (or document) of the artwork',
        reply_markup=start_processing_reply
    )


@bot.message_handler(commands=['find_similar'])
def start_processing_message(message):
    user_id = message.from_user.id

    if user_exist(user_id):

        if user_state[user_id]['image'] is None:
            bot.send_message(
                message.chat.id,
                'Please upload an image (or document)!',
                reply_markup=start_processing_reply
            )

        else:

            image = user_state[user_id]['image']
            file_name = f"{user_id}_{datetime.now()}.png"
            image.save(os.path.join(cfg.IMG_SAVING_DIR, file_name))

            try:
                with requests.post(
                        cfg.SITE_LINK_UPLOAD_FILE,
                        json={
                            'path_target_image': os.path.join(cfg.IMG_SAVING_DIR, file_name),
                        },
                ) as response:
                    if response.status_code == 200:
                        response_json = response.json()

                        send_final_response(message=message, raw_response=response_json)

                    else:
                        bot.send_message(
                            message.chat.id,
                            f"Got error in server, please try again later!",
                            reply_markup=start_processing_reply
                        )
            except:
                bot.send_message(
                    message.chat.id,
                    f"Got error in server, please try again later!",
                    reply_markup=start_processing_reply
                )

    else:
        start_message(message)


@bot.message_handler(content_types=['document', 'photo'])
def handle_image(message):
    user_id = message.from_user.id

    if user_exist(user_id):
        if message.content_type == 'document':
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            image = io.BytesIO(downloaded_file)
            image = Image.open(image)

            user_state[user_id]['image'] = image

            bot.send_message(message.chat.id, 'Image loaded!')
            bot.send_message(message.chat.id, 'Now you can use a \"/find_similar\" command to find similar artworks',
                                              reply_markup=start_processing_reply)

        elif message.content_type == 'photo':
            file_info = bot.get_file(message.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            image = io.BytesIO(downloaded_file)
            image = Image.open(image)

            user_state[user_id]['image'] = image

            bot.send_message(message.chat.id, 'Image loaded!')
            bot.send_message(message.chat.id, 'Now you can use a \"/find_similar\" command to find similar artworks',
                             reply_markup=start_processing_reply)

        else:
            bot.send_message(
                message.chat.id,
                "Something went wrong. Please try again later!",
                reply_markup=start_reply
            )

    else:
        start_message(message)


def send_final_response(message, raw_response: str):

    age = raw_response['age']
    similar_images = raw_response['similar_paths'][:max(10, user_state[message.from_user.id]['n_images'])]

    bot.send_message(message.chat.id,
                     text='Estimated year of painting: {}'.format(age))
    medias = []

    for data_i, data in enumerate(similar_images):
        photo = open(os.path.join(cfg.IMG_BASE_DIR, data[0], 'img.jpg'), 'rb')
        caption = f'by {data[2]}\nstyle: {data[1].replace("_", " ")}'
        if not math.isnan(data[3]):
            caption += f'\n{int(data[3])}'

        medias.append(types.InputMediaPhoto(photo, caption=caption))

    bot.send_media_group(message.chat.id, medias)


if __name__ == '__main__':
    from time import sleep

    print("Start to running BOT")
    for i in range(1, 0, -1):
        print(f"Bot will run after {i} seconds!")
        sleep(1)
    print("Let's go")
    bot.polling(none_stop=True)
