import os

SITE_LINK = "http://192.168.0.110:5000/"

SITE_LINK_UPLOAD_FILE = os.path.join(SITE_LINK, "api/uploadFile")
IMG_SAVING_DIR = "loaded_images"
MAIN_SITE_LINK = os.path.join("http://192.168.0.110:5000/", "")
IMG_BASE_DIR = "static/data/data"
os.makedirs(IMG_SAVING_DIR, exist_ok=True)
