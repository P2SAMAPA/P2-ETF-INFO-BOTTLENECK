import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class VariationalIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=0.001):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)
        self.beta = beta

    def forward(self, x):
        mean, logvar = self.encoder(x)
        # Reparameterisation trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        y_pred = self.decoder(z)
        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        return y_pred, kl

    def loss(self, x, y):
        y_pred, kl = self.forward(x)
        mse = nn.MSELoss()(y_pred.squeeze(), y)
        loss = mse + self.beta * kl
        return loss, mse, kl

    def predict(self, x):
        mean, _ = self.encoder(x)
        return self.decoder(mean).squeeze()
