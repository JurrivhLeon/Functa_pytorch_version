# Functa_pytorch_version
A Pytorch implement of the implicit network put forward in paper Emilien Dupont, Hyunjik Kim, SM Eslami, Danilo Rezende and Dan Rosenbaum. From data to functa: Your data point is a function and you can treat it like one. arXiv preprint arXiv:2201.12204, 2022.

To fit implicit neural representations on CIFAR-10, run ```inr_fit.py```. The model I choose is SIREN (Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein. Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems, 33, 2020.).<br>
Some results:


The codes for training modulated SIREN using meta-learning functa is in ```trainer.py```. After training, you can run ```make_functaset.py``` to fit functaset on CIFAR-10 dataset.<br>
The codes for training and evaluating classifier on CIFAR-10 functaset is in ```classifier.py``` and ```classifier_eval.py```.
