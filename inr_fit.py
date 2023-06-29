"""
Neural Network and Deep Learning, Final Project
Functa.
Junyi Liao, 20307110289
Fit INRs for images.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar_loader
from SIREN import SIREN
from utils import set_random_seeds
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from tqdm import tqdm


def fitINR(
        image,
        coord,
        device,
        hidden_features=256,
        num_layers=10,
        lr=1e-2,
        max_step=100,
        tol=1e-4,
        report=False,
):
    """
    :param image: the original image, to be fitted.
    :param coord: coordinates of pixels / meshgrid.
    :param device: running device.
    :param hidden_features: number of neurons in hidden layers in SIREN.
    :param num_layers: number of hidden layers in SIREN.
    :param lr: learning rate.
    :param max_step: number of optimizer steps.
    :param tol: min mse loss.
    :param report: whether report the result.
    :return: Fitted image and Peak Signal-to-Noise Ratio (PSNR).
    """
    coord = coord.unsqueeze(0).to(device)
    image = image.view(1, 3, -1).moveaxis(1, -1).to(device)
    inr = SIREN(hidden_features=hidden_features, num_layers=num_layers).to(device)
    optimizer = optim.SGD(inr.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)
    losses = []
    # Fit INR for the input image.
    for _ in range(int(max_step)):
        fitted = inr(coord)
        loss = criterion(fitted, image)
        losses.append(loss.item())
        if loss.item() < tol:
            break
        optimizer.zero_grad()
        loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(inr.parameters(), 1)
        # Update.
        optimizer.step()

    # Restore the image.
    fitted = inr(coord)
    # Compute the SNR.
    mse = criterion(fitted, image)
    snr = -10 * torch.log10(mse)
    if report:
        print(f'Step: {max_step}, MSE: {mse}, SNR: {snr}')
    return fitted.detach().cpu(), snr.item()


# Visualization.
def plot_fit(original, fitted100, fitted200, fname='example.png'):
    """
    :param original: original images.
    :param fitted100: images fitted within 100 steps.
    :param fitted200: images fitted within 200 steps.
    :param fname: file name (to be saved as).
    """
    os.makedirs('./fitted_results', exist_ok=True)
    figure = plt.figure(figsize=(18, 15))
    nrow, ncol = 5, 6
    for col in range(ncol):
        orig = original[col][0].view(3, 32, 32).moveaxis(0, -1).numpy()
        fit100 = fitted100[col][0].view(32, 32, 3).numpy()
        fit200 = fitted200[col][0].view(32, 32, 3).numpy()
        figure.add_subplot(nrow, ncol, col + 1)
        plt.axis("off")
        plt.imshow(orig)
        # plt.title('Original image', fontsize=8, pad=8)
        figure.add_subplot(nrow, ncol, col + ncol + 1)
        plt.axis("off")
        plt.imshow(fit100)
        # plt.title('Restored image (Step=100)', fontsize=8, pad=8)
        figure.add_subplot(nrow, ncol, col + ncol * 2 + 1)
        plt.axis("off")
        plt.imshow(np.abs(fit100 - orig) / orig.mean())
        # plt.title('Pixel-wise Error (Step=100)', fontsize=8, pad=8)
        figure.add_subplot(nrow, ncol, col + ncol * 3 + 1)
        plt.axis("off")
        plt.imshow(fit200)
        # plt.title('Restored image (Step=200)', fontsize=8, pad=8)
        figure.add_subplot(nrow, ncol, col + ncol * 4 + 1)
        plt.axis("off")
        plt.imshow(np.abs(fit200 - orig) / orig.mean())
        # plt.title('Pixel-wise Error (Step=200)', fontsize=8, pad=8)
    plt.savefig(osp.join('./fitted_results', fname))
    plt.close()


if __name__ == '__main__':
    set_random_seeds(98460)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataloader = get_cifar_loader('../data', train=True, batch_size=1)
    x, y = torch.meshgrid(torch.arange(32), torch.arange(32))
    x = x.float().view(-1).unsqueeze(0)
    y = y.float().view(-1).unsqueeze(0)
    meshgrid = torch.cat((x, y), dim=0).T
    imgs, res100, res200 = [], [], []
    for i, (img, lab) in enumerate(dataloader):
        if i >= 6:
            break
        imgs.append(img)
        restored100, psnr100 = fitINR(img, meshgrid, device, max_step=100, report=True)
        res100.append(restored100)
        restored200, psnr200 = fitINR(img, meshgrid, device, max_step=200, report=True)
        res200.append(restored200)
    plot_fit(imgs, res100, res200)

    psnr100s = []
    prog_bar100 = tqdm(dataloader)
    for i, (img, lab) in enumerate(prog_bar100):
        restored100, psnr100 = fitINR(img, meshgrid, device, max_step=100)
        psnr100s.append(psnr100)
        prog_bar100.set_description(desc=f'PSNR: {psnr100}')

    psnr200s = []
    prog_bar200 = tqdm(dataloader)
    for i, (img, lab) in enumerate(prog_bar200):
        restored200, psnr200 = fitINR(img, meshgrid, device, max_step=200)
        psnr200s.append(psnr200)
        prog_bar200.set_description(desc=f'PSNR: {psnr200}')

    np.save('fitted_results/siren_100_psnr.npy', np.array(psnr100s))
    np.save('fitted_results/siren_200_psnr.npy', np.array(psnr200s))
    # Report the average PSNR.
    print(sum(psnr100s) / len(psnr100s))
    print(sum(psnr200s) / len(psnr200s))
