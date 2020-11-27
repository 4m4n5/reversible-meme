import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MemeDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(MemeDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform
        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))
        self.mods = json.load(open(data_path + '/{}_mods.json'.format(split_type), 'r'))
        self.word_to_idx = json.load(open(data_path + '/word_dict.json'))
        self.idx_to_word =  {v: k for k, v in self.word_to_idx.items()}

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(self.mods[index])

    def __len__(self):
        return len(self.captions)
