#!/usr/bin/env python
#
# Keras GAN Implementation
# See: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
#%matplotlib inline
import os
os.environ["KERAS_BACKEND"] = "theano"
import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.layers import Input
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import cPickle, random, sys, keras
from keras.models import Model
from tqdm import tqdm
from celeba_data import celeba_npy
from keras.utils.visualize_util import plot


channel = 3
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
data = np.load(celeba_npy)
data = np.transpose(data, [0, 3, 1, 2])
X_train = data[:175000]
X_test = data[175000:]

print X_train.shape, X_test.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

shp = X_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
nch = 200
g_input = Input(shape=[100])
H = Dense(nch*14*14, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape([nch, 14, 14])(H)  # [nch, 14, 14]
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(3, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

plot(generator, to_file='model.png')


# Build Discriminative model ...
d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()


# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


def plot_loss(losses, epoch):
    #  display.clear_output(wait=True)
    #  display.display(plt.gcf())
    plt.figure(figsize=(20, 8))
    plt.plot(losses["d"], label='Discriminative Loss')
    plt.plot(losses["g"], label='Generative Loss')
    plt.legend()
    plt.savefig('figs_celeba/loss_' + str(epoch) + '.png')
    # plt.savefig('figs/loss.png')


def plot_gen(epoch, n_ex=16, dim=(4,4), figsize=(10,10)):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)

        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figs_celeba/gen_' + str(epoch) + '.png')
    # plt.savefig('figs/gen.png')

ntrain = 10000
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
XT = X_train[trainidx,:,:,:]

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])
print 'Generator predicting ... '
print noise_gen.shape
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))

n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n, 1] = 1
y[n:, 0] = 1


make_trainable(discriminator, True)
discriminator.fit(X, y, nb_epoch=1, batch_size=128)
print 'Discriminator predicting ... '
y_hat = discriminator.predict(X)


# Measure accuracy of pre-trained discriminator network
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d": [], "g": []}


# Set up our main training loop
def train_for_n(start_epoch=0, nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):
        # Make generative images
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_trainable(discriminator, False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        # Updates plots
        epoch_ind = e + start_epoch
        if epoch_ind % plt_frq == plt_frq-1:
            plot_loss(losses, epoch_ind + 1)
            plot_gen(epoch_ind + 1)


num_epoch_1 = 20000
num_epoch_2 = 10000
plt_frq = 2000

train_for_n(nb_epoch=num_epoch_1, plt_frq=plt_frq, BATCH_SIZE=32)

opt.lr.set_value(1e-5)
dopt.lr.set_value(1e-4)
train_for_n(start_epoch=num_epoch_1, nb_epoch=num_epoch_2, plt_frq=plt_frq, BATCH_SIZE=32)

opt.lr.set_value(1e-6)
dopt.lr.set_value(1e-5)
train_for_n(start_epoch=num_epoch_1 + num_epoch_2, nb_epoch=num_epoch_2, plt_frq=plt_frq, BATCH_SIZE=32)

# Plot the final loss curves
plot_loss(losses, 'Final')

# Plot some generated images from our GAN
plot_gen('Final', 25, (5, 5), (12, 12))


def plot_real(n_ex=16, dim=(4,4), figsize=(10,10) ):

    idx = np.random.randint(0,X_train.shape[0],n_ex)
    generated_images = X_train[idx, :, :, :]

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1],i+1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figs_celeba/real.png')

# Plot real images for comparison
plot_real()

