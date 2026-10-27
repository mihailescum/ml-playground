import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.encoder = nn.RNN(d_in, d_hidden, num_layers=1, nonlinearity="relu")
        self.decoder = nn.RNN(d_in, d_hidden, num_layers=1, nonlinearity="relu")
        self.output_layer = nn.Linear(d_hidden, d_out)

    def forward(self, X, y):  # X = (X_t)_t with X_t =
        _, last_hidden = self.encoder(X)
        s, _ = self.decoder()

        return self.output_layer(rnn_outputs)

    @torch.no_grad()
    def generate(self, a, max_len=100):
        # a: (1, m), one input sequence
        _, s = self.encoder(a.to(dtype=torch.float32))
        y = torch.zeros((1, self.d_in), dtype=torch.float32)
        y[0, -1] = 1.0

        out = []
        for _ in range(max_len):
            _, s = self.decoder(y, s)
            scores = self.output_layer(s)  # (1, 4)
            prediction = torch.distributions.Categorical(logits=scores).sample().item()
            if prediction == y.size()[1] - 1:
                break
            out.append(prediction)
            y[:] = 0
            y[0, prediction] = 1
        return out
