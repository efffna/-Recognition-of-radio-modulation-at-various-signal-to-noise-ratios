import os,random
import numpy as np
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU, SeparableConv2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from keras.utils import to_categorical
from keras import regularizers
from keras.layers import LSTM, Add
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization



path = '/content/drive/My Drive/signal_dataset/'
Xd = pickle.load(open(path + "RML2016.10a_dict.pkl", 'rb'), encoding='latin1')
snrs, mods = map(lambda j: sorted((set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)


np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len((yy)), max(yy) + 1])
    yy1[np.arange(len((yy))), yy] = 1
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods



dr = 0.5
model = models.Sequential()
model.add(Reshape(([1] + in_shp), input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", init='glorot_uniform',
                 data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',
                 data_format="channels_first"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), init='he_normal', name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



''' # ResNet
input_tensor= Input(shape=(in_shp))
x = layers.Reshape(([1]+in_shp))(input_tensor)
x=layers.ZeroPadding2D((0, 2))(x)
x = layers.Conv2D(64, (3, 3),padding='valid', activation="relu", name="conv1", init='glorot_uniform',data_format="channels_first")(x)
x = layers.Conv2D(32, (2, 3),padding='valid', activation="relu", name="conv2", init='glorot_uniform',data_format="channels_first")(x)
x = layers.GlobalAveragePooling2D()(x)
output_tensor = layers.Dense(11, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
'''

''' # LSTM
model=models.Sequential()
model.add(Reshape(([1]+[1]+in_shp), input_shape=in_shp))
model.add(Bidirectional(ConvLSTM2D(filters=64,kernel_size=(3, 3),padding='same',activation='relu')))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dense(len(classes), activation='softmax'))
'''

batch_size = 512
nb_epoch = 100

filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])


test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

acc = {}
for snr in snrs:


    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i_hat = model.predict(test_X_i)

    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])


    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)


print(acc)

fd = open('results_cnn2_d0.5.dat','wb')
pickle.dump( ("CNN2", 0.5, acc) , fd )
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.grid()