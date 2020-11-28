import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg19
import numpy as np


class Encoder(nn.Module):
    def __init__(self, txt_enc_dim, img_enc_dim, enc_dim, word_dict, img_enc_net='resnet18', use_glove=True, glove_path='data/glove/glove.twitter.27B.200d.txt', train_enc=True, hidden_dim=512):
        super(Encoder, self).__init__()

        # Consider last element of hidden dims as final enc dimension
        self.enc_dim = enc_dim
        # Initialize text encoder
        self.txt_encoder = TextEncoder(word_dict, txt_enc_dim, use_glove, glove_path, train_enc)
        # Initialize text encoder
        self.img_encoder = ImageEncoder(img_enc_dim, img_enc_net)

        # Design final encoding network using hidden dims
        self.gate_w = nn.Parameter(torch.rand(1))
        self.gate_fc = nn.Sequential(
            nn.Linear(img_enc_dim+txt_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.res_w = nn.Parameter(torch.rand(1))
        self.res_fc = nn.Sequential(
            nn.Linear(img_enc_dim+txt_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize seperate layers for mu and sigma
        self.fc_mu = nn.Linear(hidden_dim, self.enc_dim)
        self.fc_var = nn.Linear(hidden_dim, self.enc_dim)

    def forward(self, img, mod):
        # Encode both data
        img_enc = self.img_encoder(img)
        txt_enc = self.txt_encoder(mod)

        # Concat on the first dimension
        x = torch.cat([img_enc, txt_enc], dim=1)
        x_gate = self.gate_fc(x) * txt_enc
        x_res = self.res_fc(x)
        x = self.gate_w * x_gate + self.res_w * x_res

        # Get mu and logvar using dc layers
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # Sample z using reparameterization
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        # Sample z using mu and logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class TextEncoder(nn.Module):
    def __init__(self, word_dict, txt_enc_dim, use_glove, glove_path, train_enc):
        super(TextEncoder, self).__init__()
        # Save the dimensions required for text encoding
        self.txt_enc_dim = txt_enc_dim

        # If using glove, load the Embedding layer with glove vectors
        if use_glove:
            self.txt_enc_layer, self.vocab_size, self.glove_dim = self.get_text_encoding_layer(
                glove_path, word_dict, train_enc)
            in_dim = 200
        else:
            self.vocab_size = len(word_dict)
            self.txt_enc_layer = nn.Embedding(self.vocab_size, self.txt_enc_dim)
            in_dim = self.txt_enc_dim

        # Initialize a FC layer for transforming encoding
        self.fc = nn.Linear(in_dim, self.txt_enc_dim)

    def forward(self, x):
        # Get embeddings for the text
        x = self.txt_enc_layer(x)
        # Get mean of all mod vectors
        x = torch.mean(x, dim=1)
        # Transform to get representation
        x = self.fc(x)

        return x

    def get_weights_matrix(self, glove_path, word_dict):
        # Load glove model using the given path
        glove, glove_dim = self.load_glove_model(glove_path)
        # Get vocab size
        matrix_len = len(word_dict)
        # Initialize empty matrix
        weights_matrix = np.zeros((matrix_len, glove_dim))
        words_found = 0

        # If word is present in glove, add the embedding else random
        for i, word in enumerate(word_dict):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1

            except KeyError:
                weights_matrix[i] = np.random(scale=0.6, size=(glove_dim, ))

        assert len(word_dict) == len(weights_matrix)

        return weights_matrix

    def get_text_encoding_layer(self, glove_path, word_dict, train_enc):
        # Get weights matrix given glove vectors and word dict
        weights_matrix = self.get_weights_matrix(glove_path, word_dict)
        vocab_size, txt_enc_dim = weights_matrix.size()
        # Initialize embedding layer using the glove weights matrix
        txt_enc_layer = nn.Embedding(vocab_size, txt_enc_dim)
        txt_enc_layer.load_state_dict({'weight': weights_matrix})

        # If embeddings are not to be trained
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
    def __init__(self, img_enc_dim, img_enc_net):
        super(ImageEncoder, self).__init__()
        # Save values for which model to use and the embedding size
        self.img_enc_net = img_enc_net
        self.img_enc_dim = img_enc_dim

        # Initialize image encoder based on given input
        self.img_encoder, img_dim = self.get_image_encoder(img_enc_net)
        # Initialize a FC layer for transforming encoding
        self.fc = nn.Linear(img_dim, self.img_enc_dim)

    def forward(self, x):
        # Encode image using model
        x = self.img_encoder(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Transform to get embeddings
        x = self.fc(x)
        return x

    def get_image_encoder(self, img_enc_net):
        if img_enc_net == 'resnet18':
            img_enc = resnet18(pretrained=True)
            img_enc = nn.Sequential(*list(img_enc.children())[:-1])
            img_dim = 512
        if img_enc_net == 'resnet34':
            img_enc = resnet34(pretrained=True)
            img_enc = nn.Sequential(*list(img_enc.children())[:-1])
            img_dim = 512
        if img_enc_net == 'resnet50':
            img_enc = resnet50(pretrained=True)
            img_enc = nn.Sequential(*list(img_enc.children())[:-1])
            img_dim = 2048
        if img_enc_net == 'vgg19':
            img_enc = vgg19(pretrained=True)
            img_enc = nn.Sequential(*list(img_enc.features.children())[:-1])
            img_dim = 512

        return img_enc, img_dim
