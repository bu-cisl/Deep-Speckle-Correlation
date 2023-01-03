# Import necessary libraries and modules
from __future__ import print_function
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Import the UNet_ResNet model and the get_unet_denseblock_x2_deeper function
from UNet_ResNet import get_unet_denseblock_x2_deeper

# Import loss functions (commented out)
#from loss_function_new import total_variation_balanced_cross_entropy, balanced_cross_entropy, gaussian_loss

# Load the training and validation data
x_train = np.load('../data/x_train.npy')
y_train = np.load('../data/y_train.npy')
x_valid = np.load('../data/x_vali.npy')
y_valid = np.load('../data/y_vali.npy')

# Split the training data into a train and validation set
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.02, shuffle=True)

# Set the project name and flag for showing ground truth
proj_name = 'MNIST_x2'
show_groundtruth_flag = False

# Set the model save path and the number of epochs, batch size, and save period
save_path = 'save/lr4/'
num_epochs = 100
batch_size = 64
save_period = 10

# Set the learning rate
lr_rate = 0.001

# Use the Agg backend for Matplotlib
plt.switch_backend('agg')

# Create and compile the model
model = get_unet_denseblock_x2_deeper()
optimizer = Adam(lr=lr_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# Create a ModelCheckpoint callback to save the model at each epoch
model_checkpoint = ModelCheckpoint(save_path + proj_name + '.{epoch:02d}.hdf5',
                                   monitor='loss', verbose=2, save_best_only=False, period=save_period)

# Fit the model on the training data
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True,
                    callbacks=[model_checkpoint], validation_data=(x_valid, y_valid))

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('total_loss_4.png')
plt.close()
