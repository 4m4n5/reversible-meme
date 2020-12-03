import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, encoder, vocab_size, enc_dim, lstm_hidden_dim=256, use_tf=False):
        super(Decoder, self).__init__()
        # Use teacher forcing?
        self.use_tf = use_tf

        self.vocab_size = vocab_size
        self.enc_dim = enc_dim

        self.embedding = encoder.txt_encoder.txt_enc_layer

        self.init_h = nn.Linear(enc_dim, lstm_hidden_dim)
        self.init_c = nn.Linear(enc_dim, lstm_hidden_dim)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(lstm_hidden_dim, enc_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(lstm_hidden_dim, vocab_size)
        self.dropout = nn.Dropout()

        self.lstm = nn.LSTMCell(enc_dim + lstm_hidden_dim, lstm_hidden_dim)

    def forward(self, encoding, caption):
        # import pdb; pdb.set_trace()
        batch_size = encoding.size(0)

        h, c = self.get_init_lstm_state(encoding)
        max_timespan = max([len(cap) for cap in caption]) - 1

        prev_words = torch.zeros(batch_size, 1).long().cuda()
        if self.use_tf:
            embedding = self.embedding(caption) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, self.vocab_size).cuda()

        for t in range(max_timespan):
            # context, alpha = self.attention(encoding.unsqueeze(1).unsqueeze(1), h)
            # gate = self.sigmoid(self.f_beta(h))
            # gated_context = gate * context

            if self.use_tf and self.training:
                # lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
                lstm_input = torch.cat((embedding[:, t], encoding), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                # lstm_input = torch.cat((embedding, gated_context), dim=1)
                lstm_input = torch.cat((embedding, encoding), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            # alphas[:, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))

        return preds, h

    def get_init_lstm_state(self, encoding):
        c = self.init_c(encoding)
        c = self.tanh(c)

        h = self.init_h(encoding)
        h = self.tanh(h)

        return h, c

    def caption(self, encoding, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        # import pdb; pdb.set_trace()
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        # alphas = torch.ones(beam_size, 1, encoding.size(1))

        completed_sentences = []
        # completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(encoding)
        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            # context, alpha = self.attention(encoding, h)
            # gate = self.sigmoid(self.f_beta(h))
            # gated_context = gate * context

            lstm_input = torch.cat((embedding, encoding), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            # alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                # completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            # alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            encoding = encoding[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        # alpha = completed_sentences_alphas[idx]
        return sentence


class AlignNet(nn.Module):
    def __init__(self, lstm_hidden_dim, enc_dim):
        super(AlignNet, self).__init__()
        self.fc_mu = nn.Linear(lstm_hidden_dim, enc_dim)
        self.fc_var = nn.Linear(lstm_hidden_dim, enc_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
