"""
Neural Network and Deep Learning, Final Project.
Functa.
Junyi Liao, 20307110289
Downstream Task: Classification.
Train a classifier on the functaset.
"""

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar_functa
from utils import adjust_learning_rate, set_random_seeds, get_accuracy, Average
from tqdm import tqdm
import numpy as np
import argparse


# A classifier of MLP structure.
class Classifier(nn.Module):
    def __init__(self, width=1024, depth=3, in_features=512, num_classes=10, dropout=0.20):
        """
        :param width: number of neurons in hidden layers.
        :param depth:  number of hidden layers.
        :param in_features: number of input features.
        :param num_classes: number of classes.
        :param dropout: dropout probability.
        """
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.net = self._make_layers()

    def _make_layers(self):
        num_features = [self.in_features] + [self.width] * self.depth + [self.num_classes]
        layers = []
        for i in range(self.depth):
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(num_features[i], num_features[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_features[self.depth], num_features[self.depth + 1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(model, train_loader, optimizer, criterion, epoch):
    """
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param epoch:
    :return: Loss.
    """
    model.train()
    device = next(iter(model.parameters())).device
    losses = []
    train_score = 0
    prog_bar = tqdm(train_loader, total=len(train_loader))
    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_score += preds.argmax(dim=-1).eq(labels).sum().item()
    accuracy = train_score / len(train_loader.dataset)
    print('epoch: %d, loss: %.4f, train acc: %.3f%s' % (
        epoch, sum(losses) / len(losses), accuracy * 100, '%'
    ))
    return losses


def eval_classifier(model, val_loader, epoch):
    """
    :param model:
    :param val_loader:
    :param epoch:
    :return: Validation Accuracy.
    """
    model.eval()
    device = next(iter(model.parameters())).device
    prog_bar = tqdm(val_loader, total=len(val_loader))
    top1acc = Average()
    top5acc = Average()

    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        top1acc_batch, top5acc_batch = get_accuracy(preds, labels, top_k=(1, 5))
        top1acc.update(top1acc_batch, labels.size(0))
        top5acc.update(top5acc_batch, labels.size(0))

    print('epoch: %d, val accuracy: top1 %.2f%s, top5 %.2f%s' % (
        epoch, top1acc.avg, '%', top5acc.avg, '%'
    ))
    return top1acc.avg, top5acc.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the classifier.')
    parser.add_argument('-p', '--path', type=str, default='./models/classifier',
                        help='The path of the model to be evaluated.')
    parser.add_argument('-w', '--width', type=int, default=512,
                        help='The width of MLP.')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='The depth of MLP.')
    args = parser.parse_args()
    # Set hyperparameters.
    set_random_seeds(618)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_epochs = 160
    lr = 1e-2
    # Saving directory.
    model_dir = args.path
    os.makedirs(model_dir, exist_ok=True)
    # Load data.
    train_functaloader = get_cifar_functa(mode='train')
    val_functaloader = get_cifar_functa(mode='val')
    # Load model and optimizer.
    classifier = Classifier(width=args.width, depth=args.depth,
                            in_features=512, num_classes=10, dropout=0.20)
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    # Training.
    train_losses = []
    val_top1accs = []
    val_top5accs = []
    best_accuracy = 0
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr, num_epochs)
        losses_epo = train_classifier(
            model=classifier,
            train_loader=train_functaloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        train_losses.extend(losses_epo)
        top1acc, top5acc = eval_classifier(
            model=classifier,
            val_loader=val_functaloader,
            epoch=epoch,
        )
        val_top1accs.append(top1acc)
        val_top5accs.append(top5acc)
        # Save the best model.
        if best_accuracy < top1acc:
            best_accuracy = top1acc
            torch.save({
                'epoch': epoch,
                'state_dict': classifier.state_dict(),
                'accuracy': best_accuracy,
            }, osp.join(model_dir, 'best_classifier.pth'))

    np.save(osp.join(model_dir, 'classifier_loss.npy'), np.array(train_losses))
    np.save(osp.join(model_dir, 'classifier_acc.npy'), np.array((val_top1accs, val_top5accs)))
