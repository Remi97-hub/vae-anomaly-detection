def kl_annealing(epoch, total_epochs):
    return min(1.0, epoch / (0.3 * total_epochs))
