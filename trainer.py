"""
Neural Network and Deep Learning, Final Project
Functa.
Junyi Liao, 20307110289
Meta Learning Functa.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar_loader
from SIREN import ModulatedSIREN
from utils import adjust_learning_rate
from tqdm import tqdm
import os


# Meta-learning for functa, train a modulated SIREN for 1 epoch.
def fit(
        model,
        data_loader,
        outer_optimizer,
        outer_criterion,
        epoch_id,
        inner_steps=3,
        inner_lr=0.01,
):
    """
    :param model:
    :param data_loader:
    :param outer_optimizer:
    :param outer_criterion:
    :param epoch_id:
    :param inner_steps:
    :param inner_lr:
    :return: Loss.
    """
    # functaset = []
    losses = []
    device = next(iter(model.parameters())).device
    modul_features = model.modul_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for images, labels in prog_bar:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1).moveaxis(1, -1).to(device)
        modulators = []
        # Inner loop.
        for batch_id in range(batch_size):
            modulator = torch.zeros(modul_features, requires_grad=True).float().to(device)
            inner_optimizer = optim.SGD([modulator], lr=inner_lr)
            # Inner Optimization.
            for step in range(inner_steps):
                # Inner optimizer step.
                inner_optimizer.zero_grad()
                fitted = model(modulator)
                inner_loss = inner_criterion(fitted, images[batch_id])
                inner_loss.backward()
                # Clip the gradient.
                torch.nn.utils.clip_grad_norm_([modulator], 1)
                # Update.
                inner_optimizer.step()
            modulator.requires_grad = False
            modulators.append(modulator)

        outer_optimizer.zero_grad()
        outer_loss = torch.tensor(0).to(device).float()
        for batch_id in range(batch_size):
            modulator = modulators[batch_id]
            # Outer Optimization.
            fitted = model(modulator)
            outer_loss += outer_criterion(fitted, images[batch_id]) / batch_size
        # Outer optimizer step.
        outer_loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        outer_optimizer.step()
        losses.append(outer_loss.item())
        # functaset.extend([
        #     {'modul': modulators[j, :].detach().cpu().numpy(),
        #      'label': labels[j].item()}
        #     for j in range(batch_size)])

        prog_bar.set_description(desc='Epoch {}, loss {:.6f}'.format(epoch_id, outer_loss.item()))

    print(f'epoch: {epoch_id}, loss: {sum(losses)/ len(losses)}')
    return sum(losses) / len(losses)


if __name__ == '__main__':
    # Training Parameters.
    lr = 5e-6
    batch_size = 128
    num_epochs = 6
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataloader = get_cifar_loader('../data', train=True, batch_size=batch_size)
    modSiren = ModulatedSIREN(height=32, width=32, hidden_features=256, num_layers=10, modul_features=512)
    optimizer = optim.Adam(modSiren.parameters(), lr=lr)
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

    os.makedirs('./models', exist_ok=True)
    best_loss = float('Inf')
    for epoch in range(num_epochs):
        loss = fit(
            modSiren, dataloader, optimizer, criterion, epoch, inner_steps=3,
        )
        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch,
                        'state_dict': modSiren.state_dict(),
                        'loss': best_loss,
                        }, './models/modSiren.pth')
