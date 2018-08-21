"""
This is a quick demo of deep speckle correlation project.

Paper link: https://arxiv.org/abs/1806.04139

Author: Yunzhe Li, Yujia Xue, Lei Tian

Computational Imaging System Lab, @ ECE, Boston University

Date: 2018.08.21
"""
import matplotlib.pyplot as plt
import numpy as np

from model import get_model_deep_speckle

# model is defined in model.py
model = get_model_deep_speckle()
# pretrained_weights.hdf5 can be downloaded from the link on our GitHub project page
model.load_weights('pretrained_weights.hdf5')

# test speckle patterns. Four types of objects (E,S,8,9),
# Each object has five speckle patterns through 5 different test diffusers
speckle_E = np.load('test_data/letter_E.npy')
speckle_S = np.load('test_data/letter_S.npy')
speckle_8 = np.load('test_data/number_8.npy')
speckle_9 = np.load('test_data/number_9.npy')

# prediction
pred_speckle_E = model.predict(speckle_E, batch_size=2)
pred_speckle_S = model.predict(speckle_S, batch_size=2)
pred_speckle_8 = model.predict(speckle_8, batch_size=2)
pred_speckle_9 = model.predict(speckle_9, batch_size=2)

# plot results
plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_E[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_E[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_S[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_S[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_8[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_8[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_9[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_9[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()
