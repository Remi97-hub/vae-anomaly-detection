import torch

def vae_loss(x, recon, mu, logvar, beta):
    recon_loss = torch.mean((x - recon) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
