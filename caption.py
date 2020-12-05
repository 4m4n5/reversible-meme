import argparse
import json
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image

from dataset import pil_loader
from decoder import Decoder, AlignNet
from encoder import Encoder
from train import data_transforms
from utils import str2bool


def generate_caption(encoder, decoder, img_path, word_dict, beam_size=5, smooth=True):
    # import pdb; pdb.set_trace()
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img).unsqueeze(0)
    mod = torch.tensor([0, 254, 5554, 2343, 255, 12, 1]).unsqueeze(0)
    features = encoder(img, mod)
    features = features.expand(beam_size, features.size(1), features.size(2))
    sentence, alpha = decoder.caption(features, beam_size)

    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break

    print(sentence_tokens)
    return sentence_tokens

def load_state_dict(path):
    # original saved file with DataParallel
    state_dict = torch.load(path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict

# TODO: FIX THIS!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reversible Meme')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--enc-lr', type=float, default=1e-4, metavar='ELR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--dec-lr', type=float, default=1e-4, metavar='DLR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--aln-lr', type=float, default=1e-4, metavar='ALR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--lambda_kld', type=float, default=1.0, metavar='lKLD')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='/u/as3ek/github/reversible-meme/data',
                        help='path to data images')
    parser.add_argument('--img-enc-net', choices=['vgg19', 'resnet18', 'resnet34'], default='resnet18',
                        help='Network to use in the encoder (default: resnet18)')
    parser.add_argument('--encoder-model', type=str,
                        default='/u/as3ek/github/reversible-meme/models/run3/encoder_3.pth')
    parser.add_argument('--decoder-model', type=str,
                        default='/u/as3ek/github/reversible-meme/models/run3/decoder_3.pth')
    parser.add_argument('--aligner-model', type=str,
                        default='')
    parser.add_argument('--glove-path', type=str,
                        default='/u/as3ek/github/reversible-meme/data/glove/glove.twitter.27B.200d.txt')
    parser.add_argument('--use-tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')
    parser.add_argument('--txt-enc-dim', type=int, default=512)
    parser.add_argument('--img-enc-dim', type=int, default=512)
    parser.add_argument('--enc-dim', type=int, default=512)
    parser.add_argument('--use-glove', type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument('--train-enc', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--model-fldr', type=str, default='/u/as3ek/github/reversible-meme/models')
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--img-path', type=str,
                        default='/u/as3ek/github/reversible-meme/data/images/Bad-Luck-Brian.png')

    args = parser.parse_args()

    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocab_size = len(word_dict)

    lstm_hidden_dim = 512
    if args.use_glove:
        lstm_hidden_dim = 200

    encoder = Encoder(args.txt_enc_dim, args.enc_dim, word_dict,
                      args.img_enc_net, args.use_glove, args.glove_path, args.train_enc)
    decoder = Decoder(encoder, vocab_size, args.enc_dim, lstm_hidden_dim, use_tf=args.use_tf)

    encoder.load_state_dict(torch.load(args.encoder_model))
    decoder.load_state_dict(torch.load(args.decoder_model))

    encoder.eval()
    decoder.eval()

    generate_caption(encoder, decoder, args.img_path, word_dict)
