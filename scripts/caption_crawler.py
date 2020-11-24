import torch
import json
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import urllib.request
import requests
import shutil
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Caption Crawler')
parser.add_argument('--sm', action='store', dest='sm', default=1)
parser.add_argument('--em', action='store', dest='em', default=1)
parser.add_argument('--ncp', action='store', dest='ncp', default=200)

base_url = "https://memegenerator.net/memes/popular/alltime/page/"

args = parser.parse_args()

start_meme = int(args.sm)
end_meme = int(args.em)
num_caption_pages = int(args.ncp)

print(args)


def download_web_image(url, name):
    r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
    full_name = '/u/as3ek/github/reversible-meme/data/images' + str(name) + '.png'
    if r.status_code == 200:
        with open(full_name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        print('Could not download ' + str(name))

    return full_name


data = pd.read_csv('/u/as3ek/github/reversible-meme/data/meme_metadata.csv')
data = data[start_meme:end_meme]

overall_data = {}

for index, row in tqdm(data.iterrows()):
    caption_count = 0
    caption_data = {}

    meme_link = row['link']
    img_link = row['img_url']
    img_name = row['name']
    local_link = download_web_image(img_link, meme_link)
    base_url = 'https://memegenerator.net'

    for page in range(1, num_caption_pages):
        url = base_url + row['link'] + '/images/popular/alltime/page/' + str(page)
        time.sleep(random.uniform(0, 0.3))
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text)

        for block in soup.findAll('div', {'class': 'single-generator-body'}):
            meme = block.find('span', {'class': 'generator-img'})
            caption = meme.find('div', {'class': 'optimized-instance-container'})
            text0 = caption.find('div', {'class': 'optimized-instance-text0'}).text.strip()
            text1 = caption.find('div', {'class': 'optimized-instance-text1'}).text.strip()
            upvotes = int(block.find('div', {'class': 'score'}).text.replace(',', '').strip())
            tmp = {
                'name': img_name,
                'link': meme_link,
                'caption_top': text0,
                'caption_bottom': text1,
                'local_link': local_link,
                'upvotes': upvotes
            }

            caption_data[caption_count] = tmp
            caption_count = caption_count + 1

    overall_data[meme_link] = caption_data

    filename = '/u/as3ek/github/reversible-meme/data/caption_data_' + \
        str(start_meme) + '_' + str(end_meme) + '.json'

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(overall_data, f, ensure_ascii=False, indent=4)
