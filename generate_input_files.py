from tqdm import tqdm
import random
import pandas as pd
import argparse
import json
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import os
nltk.download('punkt')

def get_synonym(word, n=3):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if len(synonyms) >= n:
                synonyms.append(word)
                return random.choice(synonyms)

def generate_json_data(metadata_path, caption_path, imgs_path, max_captions_per_image,
                       min_word_count, train_perc, max_caption_length,
                       data_folder):
    # Read the metadata csv
    metadata = pd.read_csv(metadata_path)

    # Read the json data
    with open(caption_path) as f:
        data = json.load(f)

    # Init counter
    word_count = Counter()

    # For storing paths
    train_img_paths = []
    train_caption_tokens = []
    train_mod_tokens = []

    val_img_paths = []
    val_caption_tokens = []
    val_mod_tokens = []

    max_length = 0

    for img in tqdm(metadata['link']):
        caption_count = 0
        if img in data:
            for key, value in data[img].items():
                # Get the caption
                sentence = ' '.join([value['caption_top'], value['caption_bottom']])

                # Split into tokens
                tokens = word_tokenize(sentence)
                tokens = [token.lower() for token in tokens][:max_caption_length]

                # TODO: Use advanced tokens
                mod_tokens = word_tokenize(value['name']) + tokens
                mod_tokens = [token.lower() for token in mod_tokens]

                keep_mod_tokens = []
                for w, tag in nltk.pos_tag(mod_tokens):
                    if tag in ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'PDT', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        keep_mod_tokens.append(get_synonym(w))

                mod_tokens = keep_mod_tokens[:max_caption_length]

                # Get image path
                img_path = value['local_link']

                if caption_count < max_captions_per_image:
                    caption_count += 1
                else:
                    break

                r = random.randint(0, 100)
                if r < int(train_perc):
                    train_img_paths.append(img_path)
                    train_caption_tokens.append(tokens)
                    train_mod_tokens.append(mod_tokens)
                else:
                    val_img_paths.append(img_path)
                    val_caption_tokens.append(tokens)
                    val_mod_tokens.append(mod_tokens)

                # Get max length of a caption to pad accordingly later
                max_length = max(max_length, len(tokens))
                word_count.update(tokens)

    words = [word for word in word_count.keys() if word_count[word] >= args.min_word_count]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}

    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3

    # Write word dict to file
    with open(data_folder + 'word_dict.json', 'w') as f:
        json.dump(word_dict, f)

    # Captions
    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_length)
    val_captions = process_caption_tokens(val_caption_tokens, word_dict, max_length)

    # Modifiers
    train_mods = process_caption_tokens(train_mod_tokens, word_dict, max_length)
    val_mods = process_caption_tokens(val_mod_tokens, word_dict, max_length)

    # Images
    with open(data_folder + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_folder + '/val_img_paths.json', 'w') as f:
        json.dump(val_img_paths, f)

    # Captions
    with open(data_folder + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_folder + '/val_captions.json', 'w') as f:
        json.dump(val_captions, f)

    # Modifiers
    with open(data_folder + '/train_mods.json', 'w') as f:
        json.dump(train_mods, f)
    with open(data_folder + '/val_mods.json', 'w') as f:
        json.dump(val_mods, f)


def process_caption_tokens(caption_tokens, word_dict, max_length):
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>']
                      for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))
    return captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files')
    parser.add_argument('--metadata-path', type=str, default='data/meme_metadata.csv')
    parser.add_argument('--caption-path', type=str, default='data/caption_data.json')
    parser.add_argument('--imgs-path', type=str, default='data/images/')
    parser.add_argument('--max-captions', type=int, default=1000,
                        help='maximum number of captions per image')
    parser.add_argument('--min-word-count', type=int, default=5,
                        help='minimum number of occurences of a word to be included in word dictionary')
    parser.add_argument('--train-ratio', type=int, default=80)
    parser.add_argument('--max-caption-length', type=int, default=20)
    parser.add_argument('--data-folder', type=str, default='data/')

    args = parser.parse_args()

    generate_json_data(args.metadata_path, args.caption_path, args.imgs_path,
                       args.max_captions, args.min_word_count, args.train_ratio,
                       args.max_caption_length, args.data_folder)
