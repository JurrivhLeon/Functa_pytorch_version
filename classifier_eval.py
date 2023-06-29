"""
Neural Network and Deep Learning, Final Project.
Functa.
Junyi Liao, 20307110289
Downstream Task: Classification.
Evaluation the performance of functaset.
"""

import torch
import argparse
from classifier import Classifier, eval_classifier
from dataloader import get_cifar_functa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the classifier.')
    parser.add_argument('-p', '--path', type=str, default='./models/classifier/best_classifier.pth',
                        help='The path of the model to be evaluated.')
    parser.add_argument('-w', '--width', type=int, default=512,
                        help='The width of MLP.')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='The depth of MLP')
    args = parser.parse_args()
    model_path = args.path
    # Set hyperparameters.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Load data.
    test_functaloader = get_cifar_functa(mode='test')
    # Load model.
    classifier = Classifier(width=args.width, depth=args.depth, in_features=512, num_classes=10, dropout=0.20)
    best_model = torch.load(model_path)
    classifier.load_state_dict(best_model['state_dict'])
    eval_classifier(classifier, test_functaloader, 0)
