import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.encoder = nn.RNN(
            d_in, d_hidden, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.decoder = nn.RNN(
            d_in, d_hidden, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.output_layer = nn.Linear(d_hidden, d_out)

    def forward(self, X, x_lens, y):
        X = F.one_hot(X, self.d_in).to(dtype=torch.float32)
        _, last_hidden = self.encoder(
            pack_padded_sequence(X, x_lens, batch_first=True, enforce_sorted=False)
        )
        s, _ = self.decoder(
            F.one_hot(y, self.d_out).to(dtype=torch.float32), last_hidden
        )
        return self.output_layer(s)

    @torch.no_grad()
    def predict(self, a, max_len=100):
        # a: (1, m), one input sequence
        _, s = self.encoder(F.one_hot(a, self.d_in).to(dtype=torch.float32))
        y = torch.zeros((1, self.d_out), dtype=torch.float32)
        y[0, -2] = 1.0

        out = []
        for _ in range(max_len):
            _, s = self.decoder(y, s)
            scores = self.output_layer(s)
            prediction = torch.distributions.Categorical(logits=scores).sample().item()
            if prediction == self.d_out - 1:
                break
            out.append(prediction)
            y[:] = 0
            y[0, prediction] = 1
        return out
