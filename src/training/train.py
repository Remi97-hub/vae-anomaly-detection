from src.training.loss import vae_loss
from src.training.annealing import kl_annealing

def train(model, loader, optimizer, epochs, device):
    model.train()

    for epoch in range(epochs):
        beta = kl_annealing(epoch, epochs)

        for (x,) in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss = vae_loss(x, recon, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
