import shutil
import requests
from bs4 import BeautifulSoup
import os


class ParserGermanprints:

    def __init__(
            self,
            headers: dict,
            url_template: str = 'http://germanprints.ru/data/authors/{}/index.php?show=all&page={}'
    ):
        self.headers = headers
        self.url_template = url_template
        self.count = 0

    def get_html(self, url: str):
        print('URL', url)
        r = requests.get(url=url, headers=HEADERS)
        return r

    def parse_page(self, url: str, path_to_save_folder: str):

        html = self.get_html(url)

        print(html.status_code)

        if html.status_code == 200:
            soup = BeautifulSoup(html.text, 'html.parser')

            divs = soup.find_all('div', class_='iteml')

            for div in divs:
                a = div.find('img')

                img_href = 'http://germanprints.ru' + a['src']

                big_img_href = img_href.replace('_01.jpg', '_03.jpg')

                res = requests.get(big_img_href, stream=True)
                with open(os.path.join(path_to_save_folder, str(self.count) + '.jpg'), 'wb') as f:
                    self.count += 1
                    shutil.copyfileobj(res.raw, f)

    def parse(
            self,
            artist: str,
            num_pages: int,
            path_to_save_folder: str
    ):

        os.makedirs(path_to_save_folder, exist_ok=True)

        for page in range(1, num_pages + 1):
            print('Page: ', page)

            url = self.url_template.format(artist, page)
            self.parse_page(url=url, path_to_save_folder=path_to_save_folder)


if __name__ == '__main__':

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }

    parser = ParserGermanprints(headers=HEADERS)

    author_name = 'walter'
    path_to_save_folder = '/home/juravlik/PycharmProjects/ssu_artworks/static/data/germanprints/{}'.format(author_name)

    parser.parse(artist=author_name,
                 num_pages=1,
                 path_to_save_folder=path_to_save_folder)
