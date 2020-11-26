import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg19
import gensim
import numpy as np


class TextEncoder(nn.Module):
    def __init__(self, word_dict, txt_enc_dim, use_glove=True, glove_path='data/glove/', train_enc=True):
        super(TextEncoder, self).__init__()
        
        self.txt_enc_dim = txt_enc_dim

        if use_glove:
            weights_matrix = self.get_weights_matrix(glove_path, word_dict)
            self.txt_enc_layer, self.vocab_size, self.glove_dim = self.get_text_encoding_layer(
                weights_matrix, train_enc)
        else:
            self.vocab_size = len(word_dict)
            self.txt_enc_layer = nn.Embedding(vocab_size, txt_enc_dim)

        # Initialize a FC layer for transforming encoding
        self.fc = nn.Linear(200, self.txt_enc_dim)

    def forward(self, x):
        # TODO: Fix this to get avgs
        x = self.txt_enc_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_weights_matrix(self, glove_path, word_dict):
        glove, glove_dim = self.load_glove_model(glove_path)
        matrix_len = len(word_dict)
        weights_matrix = np.zeros((matrix_len, glove_dim))
        words_found = 0

        for i, word in enumerate(word_dict):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1

            except KeyError:
                weights_matrix[i] = np.random(scale=0.6, size=(glove_dim, ))

        assert len(word_dict) == len(weights_matrix)

        return weights_matrix

    def get_text_encoding_layer(self, weights_matrix, train_enc):
        vocab_size, txt_enc_dim = weights_matrix.size()
        txt_enc_layer = nn.Embedding(vocab_size, txt_enc_dim)
        txt_enc_layer.load_state_dict({'weight': weights_matrix})

        if train_enc == False:
            emb_layer.weight.requires_grad = False

        return txt_enc_layer, vocab_size, txt_enc_dim

    def load_glove_model(self, glove_path):
        print("Loading Glove Model")
        f = open(glove_path, 'r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel), " words loaded!")
        return gloveModel, len(wordEmbedding)


class ImageEncoder(nn.Module):
    def __init__(self, img_enc_net='resnet18', img_enc_dim):
        super(ImageEncoder, self).__init__()
        self.img_enc_net = img_enc_net
        self.img_enc_dim = img_enc_dim

        # Initialize image encoder based on given input
        self.img_encoder, img_dim = self.get_img_encoder(img_enc_net)
        # Initialize a FC layer for transforming encoding
        self.fc = nn.Linear(img_dim, img_enc_dim)

    def forward(self, x):
        x = self.img_enc_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_image_encoder(self, img_enc_net):
        if img_enc_net == 'resnet18':
            img_enc = resnet18(pretrained=True)
            img_enc = nn.Sequential(*list(self.net.children())[:-2])
            img_dim = 512
        if img_enc_net == 'resnet34':
            img_enc = resnet34(pretrained=True)
            img_enc = nn.Sequential(*list(self.net.children())[:-2])
            img_dim = 512
        if img_enc_net == 'resnet50':
            img_enc = resnet50(pretrained=True)
            img_enc = nn.Sequential(*list(self.net.children())[:-2])
            img_dim = 2048
        if img_enc_net == 'vgg19':
            img_enc = vgg19(pretrained=True)
            img_enc = nn.Sequential(*list(self.net.features.children())[:-1])
            img_dim = 512

        return img_enc, img_dim
