"""
Neural Network and Deep Learning, Final Project.
Functa.
Junyi Liao, 20307110289
Create functaset of latent vectors.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar_loader
from SIREN import ModulatedSIREN
from utils import set_random_seeds
import joblib
from tqdm import tqdm


# Create a functaset on MNIST or CIFAR-10.
def create_functaset(
        model,
        data_loader,
        inner_steps=100,
        inner_lr=0.01,
):
    """
    :param model:
    :param data_loader:
    :param inner_steps:
    :param inner_lr:
    """
    assert data_loader.batch_size == 1
    functaset = []
    device = next(iter(model.parameters())).device
    modul_features = model.modul_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for image, label in prog_bar:
        image = image[0].view(3, -1).T.to(device)
        modulator = torch.zeros(modul_features, requires_grad=True).float().to(device)
        inner_optimizer = optim.SGD([modulator], lr=inner_lr)
        mse = 0
        # Inner Optimization.
        for step in range(inner_steps):
            fitted = model(modulator)
            inner_loss = inner_criterion(fitted, image)
            mse = inner_loss.item()

            # Inner optimizer step.
            inner_optimizer.zero_grad()
            inner_loss.backward()
            # Clip the gradient.
            torch.nn.utils.clip_grad_norm_([modulator], 1)
            # Update.
            inner_optimizer.step()
        prog_bar.set_description(f'MSE: {mse}')

        functaset.append(
            {'modul': modulator.detach().cpu().numpy(),
             'label': label[0].item()})

    return functaset


# Split the train, validation and test functaset.
def split(functaset):
    assignment = torch.tensor([0] * 35000 + [1] * 5000 + [2] * 10000)
    assignment = assignment[torch.randperm(50000)]
    train_set, val_set, test_set = [], [], []
    for i in range(50000):
        if assignment[i] == 0:
            train_set.append(functaset[i])
        elif assignment[i] == 1:
            val_set.append(functaset[i])
        else:
            test_set.append(functaset[i])

    os.makedirs('./functaset')
    joblib.dump(train_set, './functaset/cifar10_train.pkl')
    joblib.dump(val_set, './functaset/cifar10_val.pkl')
    joblib.dump(test_set, './functaset/cifar10_test.pkl')


if __name__ == '__main__':
    set_random_seeds(2023)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataloader = get_cifar_loader('../data', train=True, batch_size=1)
    modSiren = ModulatedSIREN(height=32, width=32, hidden_features=256, num_layers=10, modul_features=512)
    pretrained = torch.load('./models/modSiren.pth')
    modSiren.load_state_dict(pretrained['state_dict'])
    functa_set = create_functaset(modSiren, dataloader, inner_steps=100, inner_lr=0.01)
    split(functa_set)
