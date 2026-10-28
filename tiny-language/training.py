import torch


class Trainer:
    def __init__(self, model, loss_fn=None, optim=None):
        self.model = model

        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if optim:
            self.optim = optim
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    def fit_epoch(self, dataloader, gradient_clip_val=0):
        epoch_loss = 0.0
        for X, x_lens, dec_in, target in dataloader:
            self.optim.zero_grad()
            logits = self.model(X, x_lens, dec_in)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            if gradient_clip_val > 0:
                params = [p for p in self.model.parameters() if p.requires_grad]
                norm = sum(torch.sum((p.grad**2)) for p in params)
                if norm > gradient_clip_val:
                    for param in params:
                        param.grad[:] *= gradient_clip_val / norm

            self.optim.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def fit(self, dataloader, gradient_clip_val=0, num_epochs=1000):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}: {self.fit_epoch(dataloader, gradient_clip_val)}")
