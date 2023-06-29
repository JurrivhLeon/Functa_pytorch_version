"""
Neural Network and Deep Learning, Final project.
Functa.
Junyi Liao, 20307110289
MLP Classifier Visualization Module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os.path as osp

ckpt_dirs512 = [
    './models/classifier_2_512',
    './models/classifier_3_512',
    './models/classifier_4_512',
]

ckpt_dirs1024 = [
    './models/classifier_2_1024',
    './models/classifier_3_1024',
    './models/classifier_4_1024',
]

epochs = 160

losses512mean = []
losses512CI = []
valAcc512 = []
losses1024mean = []
losses1024CI = []
valAcc1024 = []

for i in range(3):
    loss512 = np.load(osp.join(ckpt_dirs512[i], 'classifier_loss.npy')).reshape(epochs, -1)
    loss512mean = np.mean(loss512, axis=-1)
    loss512CI = np.quantile(loss512, (0.025, 0.975), axis=-1)
    losses512mean.append(loss512mean)
    losses512CI.append(loss512CI)
    valAcc512.append(np.load(osp.join(ckpt_dirs512[i], 'classifier_acc.npy'))[0, :])
    loss1024 = np.load(osp.join(ckpt_dirs1024[i], 'classifier_loss.npy')).reshape(epochs, -1)
    loss1024mean = np.mean(loss1024, axis=-1)
    loss1024CI = np.quantile(loss1024, (0.025, 0.975), axis=-1)
    losses1024mean.append(loss1024mean)
    losses1024CI.append(loss1024CI)
    valAcc1024.append(np.load(osp.join(ckpt_dirs1024[i], 'classifier_acc.npy'))[0, :])

plt.figure()
plt.plot(np.arange(0, epochs), losses512mean[0], linestyle='solid', color='gold', linewidth=1.2)
plt.fill_between(np.arange(0, epochs),
                 losses512CI[0][0, :], losses512CI[0][1, :],
                 color='khaki', alpha=0.5)
plt.plot(np.arange(0, epochs), losses512mean[1], linestyle='solid', color='cornflowerblue', linewidth=1.2)
plt.fill_between(np.arange(0, epochs),
                 losses512CI[1][0, :], losses512CI[1][1, :],
                 color='lightskyblue', alpha=0.5)
plt.plot(np.arange(0, epochs), losses512mean[2], linestyle='solid', color='mediumseagreen', linewidth=1.2)
plt.fill_between(np.arange(0, epochs),
                 losses512CI[2][0, :], losses512CI[2][1, :],
                 color='palegreen', alpha=0.5)
plt.xlabel('Epochs')
plt.yticks(np.arange(0, 2.55, 0.25))
plt.legend([r'2 Layers, Mean', r'2 Layers, 95% CI',
            r'3 Layers, Mean', r'3 Layers, 95% CI',
            r'4 Layers, Mean', r'4 Layers, 95% CI'])
plt.title('The Loss of MLPs of width 512', fontsize=12, pad=10)
plt.show()

plt.figure()
plt.plot(np.arange(0, epochs), valAcc512[0], linestyle='solid', color='gold', linewidth=1.2)
plt.plot(np.arange(0, epochs), valAcc512[1], linestyle='solid', color='cornflowerblue', linewidth=1.2)
plt.plot(np.arange(0, epochs), valAcc512[2], linestyle='solid', color='mediumseagreen', linewidth=1.2)
plt.xlabel('Epochs')
plt.yticks(np.arange(0, 101, 10))
plt.legend([r'2 Layers', r'3 Layers', r'4 Layers'])
plt.title('The Validation Accuracy of MLPs of width 512', fontsize=12, pad=10)
plt.show()

plt.figure()
plt.plot(np.arange(0, epochs), losses1024mean[0], linestyle='solid',
         color='gold', linewidth=1.2)
plt.fill_between(np.arange(0, epochs), losses1024CI[0][0, :], losses1024CI[0][1, :],
                 color='khaki', alpha=0.5)
plt.plot(np.arange(0, epochs), losses1024mean[1], linestyle='solid',
         color='cornflowerblue', linewidth=1.2)
plt.fill_between(np.arange(0, epochs), losses1024CI[1][0, :], losses1024CI[1][1, :],
                 color='lightskyblue', alpha=0.5)
plt.plot(np.arange(0, epochs), losses1024mean[2], linestyle='solid',
         color='mediumseagreen', linewidth=1.2)
plt.fill_between(np.arange(0, epochs), losses1024CI[2][0, :], losses1024CI[2][1, :],
                 color='palegreen', alpha=0.5)
plt.xlabel('Epochs')
plt.yticks(np.arange(0, 2.55, 0.25))
plt.legend([r'2 Layers, Mean', r'2 Layers, 95% CI',
            r'3 Layers, Mean', r'3 Layers, 95% CI',
            r'4 Layers, Mean', r'4 Layers, 95% CI'])
plt.title('The Loss of MLPs of width 1024', fontsize=12, pad=10)
plt.show()

plt.figure()
plt.plot(np.arange(0, epochs), valAcc1024[0], linestyle='solid', color='gold', linewidth=1.2)
plt.plot(np.arange(0, epochs), valAcc1024[1], linestyle='solid', color='cornflowerblue', linewidth=1.2)
plt.plot(np.arange(0, epochs), valAcc1024[2], linestyle='solid', color='mediumseagreen', linewidth=1.2)
plt.xlabel('Epochs')
plt.yticks(np.arange(0, 101, 10))
plt.legend([r'2 Layers', r'3 Layers', r'4 Layers'])
plt.title('The Validation Accuracy of MLPs of width 1024', fontsize=12, pad=10)
plt.show()
