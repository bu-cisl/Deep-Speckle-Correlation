import matplotlib.pyplot as plt
import numpy as np

from model import get_model_deep_speckle

model = get_model_deep_speckle()
model.load_weights('pretrained_weights.hdf5')

speckle_E = np.load('test_data/letter_E.npy')
speckle_S = np.load('test_data/letter_S.npy')
speckle_8 = np.load('test_data/number_8.npy')
speckle_9 = np.load('test_data/number_9.npy')

pred_speckle_E = model.predict(speckle_E, batch_size=2)
pred_speckle_S = model.predict(speckle_S, batch_size=2)
pred_speckle_8 = model.predict(speckle_8, batch_size=2)
pred_speckle_9 = model.predict(speckle_9, batch_size=2)

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
