import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import MemeDataset
from decoder import Decoder, AlignNet
from encoder import Encoder, TextEncoder, ImageEncoder
from utils import AverageMeter, accuracy, calculate_caption_lengths, str2bool
import os

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main(args):
    # writer = SummaryWriter()
    print(args)
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocab_size = len(word_dict)

    lstm_hidden_dim = 256
    if args.use_glove:
        lstm_hidden_dim = 200

    encoder = Encoder(args.txt_enc_dim, args.img_enc_dim, args.enc_dim, word_dict,
                      args.img_enc_net, args.use_glove, args.glove_path, args.train_enc)
    decoder = Decoder(encoder, vocab_size, args.enc_dim, lstm_hidden_dim, use_tf=args.use_tf)
    aligner = AlignNet(lstm_hidden_dim, enc_dim=args.enc_dim)

    if args.encoder_model:
        encoder.load_state_dict(torch.load(args.encoder_model))

    if args.decoder_model:
        decoder.load_state_dict(torch.load(args.decoder_model))

    if args.aligner_model:
        aligner.load_state_dict(torch.load(args.aligner_model))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        aligner = nn.DataParallel(aligner)

    encoder.cuda()
    decoder.cuda()
    aligner.cuda()

    enc_optim = optim.Adam(encoder.parameters(), lr=args.enc_lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=args.dec_lr)
    aln_optim = optim.Adam(aligner.parameters(), lr=args.aln_lr)

    scheduler_enc = optim.lr_scheduler.StepLR(enc_optim, args.step_size)
    scheduler_dec = optim.lr_scheduler.StepLR(dec_optim, args.step_size)
    scheduler_aln = optim.lr_scheduler.StepLR(aln_optim, args.step_size)

    cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    train_loader = torch.utils.data.DataLoader(
        MemeDataset(data_transforms, args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        MemeDataset(data_transforms, args.data, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    for epoch in range(1, args.epochs + 1):
        scheduler_enc.step()
        scheduler_dec.step()
        scheduler_aln.step()
        train(epoch, encoder, decoder, aligner, enc_optim, dec_optim, aln_optim,
              cross_entropy_loss, train_loader, word_dict, args.lambda_kld, args.log_interval)
        # validate(epoch, encoder, decoder, cross_entropy_loss, val_loader,
        #          word_dict, args.alpha_c, args.log_interval, writer)

        # Make subdireactory if not exists
        dir = os.path.join(args.model_fldr, args.id)
        if not os.path.exists(dir):
            os.makedirs(dir)

        enc_file = os.path.join(dir, 'encoder_' + str(epoch) + '.pth')
        dec_file = os.path.join(dir, 'decoder_' + str(epoch) + '.pth')
        aln_file = os.path.join(dir, 'aligner_' + str(epoch) + '.pth')

        torch.save(encoder.module.state_dict(), enc_file)
        torch.save(decoder.module.state_dict(), dec_file)
        torch.save(aligner.module.state_dict(), aln_file)

        print('Saved Model!')


def train(epoch, encoder, decoder, aligner, enc_optim, dec_optim, aln_optim, cross_entropy_loss, train_loader, word_dict, lambda_kld, log_interval):
    # import pdb; pdb.set_trace()
    encoder.train()
    decoder.train()
    aligner.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (img, mod, cap) in enumerate(train_loader):
        img, mod, cap = Variable(img).cuda(), Variable(mod).cuda(), Variable(cap).cuda()

        enc_optim.zero_grad()
        dec_optim.zero_grad()
        aln_optim.zero_grad()

        enc_features, mu, logvar = encoder(img, mod)
        preds, h = decoder(enc_features, cap)

        targets = cap[:, 1:]
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        # Captioning Cross Entripy loss
        captioning_loss = cross_entropy_loss(preds, targets)

        # KL Divergence of latent space with normal distribution
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        # Total loss
        loss = captioning_loss + lambda_kld*kld_loss
        loss.backward()
        enc_optim.step()
        dec_optim.step()

        # Alignment loss comparing the two distributions
        logvar = logvar.detach()
        mu = mu.detach()
        z_hat, mu_hat, logvar_hat = aligner(h.detach())

        rev_kld_loss = torch.mean(torch.sum(torch.log(torch.sqrt(logvar.exp(
        ) / logvar_hat.exp())) + (logvar_hat.exp() + (mu_hat - mu)**2) / 2*logvar.exp(), dim=1) - 0.5, dim=0)

        rev_kld_loss.backward()
        aln_optim.step()

        total_caption_length = calculate_caption_lengths(word_dict, cap)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_idx, len(train_loader), loss=losses, top1=top1, top5=top5))
    # writer.add_scalar('train_loss', losses.avg, epoch)
    # writer.add_scalar('train_top1_acc', top1.avg, epoch)
    # writer.add_scalar('train_top5_acc', top5.avg, epoch)


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
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='/u/as3ek/github/reversible-meme/data',
                        help='path to data images')
    parser.add_argument('--img-enc-net', choices=['vgg19', 'resnet18', 'resnet34'], default='resnet18',
                        help='Network to use in the encoder (default: resnet18)')
    parser.add_argument('--encoder-model', type=str, help='path to model')
    parser.add_argument('--decoder-model', type=str, help='path to model')
    parser.add_argument('--aligner-model', type=str, help='path to model')
    parser.add_argument('--glove-path', type=str,
                        default='/u/as3ek/github/reversible-meme/data/glove/glove.twitter.27B.200d.txt')
    parser.add_argument('--use-tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')
    parser.add_argument('--txt-enc-dim', type=int, default=256)
    parser.add_argument('--img-enc-dim', type=int, default=256)
    parser.add_argument('--enc-dim', type=int, default=256)
    parser.add_argument('--use-glove', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--train-enc', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--model-fldr', type=str, default='/u/as3ek/github/reversible-meme/models')
    parser.add_argument('--id', type=str, default='no_glove_rerun')

    main(parser.parse_args())
