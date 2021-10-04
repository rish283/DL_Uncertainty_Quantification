# IMPORT NECESSARY PYTHON LIBRARIES
import tensorflow as tf
from keras import backend as K
import keras.layers
from tensorflow.keras.layers import Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Dropout, AveragePooling2D, Conv2D, Input
import numpy as np
from sklearn.utils import shuffle
from tqdm.keras import TqdmCallback
import time
import os
from scipy.signal import decimate as dc

# NAMES OF VARIOUS TRAINED UQ MODELS
qt = 'md_kmni.h5'
stll = 'svi_ll_kmni.h5'
st = 'svi_kmni.h5'
enss = 'kmnist/'

# LOCATION OF FINETUNED MODEL
pt = os.path.join('Models/finetuned/', qt)

#IF MAIN MODEL TRAINING IS ENABLED, mdtr = 1
mdtr = 0

#IF SVI MODEL TRAINING IS ENABLED, svlltr, svtr = 1
svlltr = 0
svtr = 0

#IF ENSEMBLE TRAINING IS ENABLED, enstr = 1
ens_tr = 0

svill_pt = os.path.join('Models/finetuned/', stll)
svi_pt = os.path.join('Models/finetuned/', st)
ens1 = 'Models/finetuned/ensemble/' + enss


class uncq:

    def mcdrop(model, dpr, itr, test_data, nop):

        if mdtr == 0:
            model.load_weights(pt)

        def PermaDropout(rate):
            return Lambda(lambda x: K.dropout(x, level=rate))

        # Store the fully connected layers
        fc = []
        for i in range(len(model.layers)):
            fc.append(model.layers[i])

        x = Dropout(dpr)(fc[1].output, training=True)
        x = fc[2](x)
        x = fc[3](x)
        x = fc[4](x)
        x = Dropout(dpr)(x, training=True)
        x = fc[5](x)
        x = Dropout(dpr)(x, training=True)
        x = fc[6](x)
        x = Dropout(dpr)(x, training=True)
        x = fc[7](x)
        predictors = fc[8](x)
        # for g in range(3, len(model.layers)-4):
        #     x = fc[g](x)
        #     x = PermaDropout(dpr)(x)
        #
        # x = fc[-4](x)
        # x = PermaDropout(dpr)(x)
        # x = fc[-2](x)
        # predictors = fc[-1](x)

        # Create a new model
        model2 = Model(inputs=model.input, outputs=predictors)

        if mdtr == 0:
            model2.load_weights(pt)

        model2.summary()
        print('MCD')

        xtest = test_data

        mcpred = []
        pr = model2.predict(xtest)

        wts = model2.layers[-2].get_weights()[0]

        pr1 = pr
        aa1 = np.max(pr, 1)
        for i in range(len(aa1)):
            for j in range(nop):
                if pr[i, j] == aa1[i]:
                    pr1[i, j] = 1
                else:
                    pr1[i, j] = 0

        start = time.time()
        for k in range(itr):
            mcpred.append(model2.predict(xtest))

            print(k)

        stop = time.time()

        duration = stop - start
        print(duration)

        # return mcpred

        stq = np.zeros((len(xtest), nop))
        mcq = np.zeros((itr, nop))

        stqm = np.zeros(len(xtest))

        for p in range(len(xtest)):
            for q in range(itr):
                # mcp[j, i] = mcpred[j][i, 0]
                # mcq[q, p] = mcpred[q][p, 0]
                mcq[q, :] = mcpred[q][p][:]

            stq[p, :] = np.std(mcq, 0)
            # stqm[p] = np.mean(mcq, 1)

        sm = np.sum(stq*pr1, 1)

        return sm, stq

    def mcdrop_ll(model, dpr, itr, test_data, nop):

        if mdtr == 0:
            model.load_weights(pt)

        def PermaDropout(rate):
            return Lambda(lambda x: K.dropout(x, level=rate))

        fc = []
        for i in range(len(model.layers)):
            fc.append(model.layers[i])

        x = Dropout(dpr)(fc[5].output, training=True)
        x = fc[6](x)
        x = fc[7](x)
        predictors = fc[8](x)

        # Create a new model
        model2 = Model(inputs=model.input, outputs=predictors)

        if mdtr == 0:
            model.load_weights(pt)

        model2.summary()
        print('MCD_LL')

        xtest = test_data

        pr = model2.predict(xtest)
        pr1 = pr
        aa1 = np.max(pr, 1)
        for i in range(len(aa1)):
            for j in range(nop):
                if pr[i, j] == aa1[i]:
                    pr1[i, j] = 1
                else:
                    pr1[i, j] = 0

        mcpred = []

        start = time.time()
        for k in range(itr):
            mcpred.append(model2.predict(xtest))

            print(k)

        stop = time.time()

        duration = stop - start
        print(duration)

        # return mcpred

        stq = np.zeros((len(xtest), nop))
        mcq = np.zeros((itr, nop))

        stqm = np.zeros(len(xtest))

        for p in range(len(xtest)):
            for q in range(itr):
                # mcp[j, i] = mcpred[j][i, 0]
                # mcq[q, p] = mcpred[q][p, 0]
                mcq[q, :] = mcpred[q][p][:]

            stq[p, :] = np.std(mcq, 0)
            # stqm[p] = np.mean(mcq, 1)

        sm = np.sum(stq*pr1, 1)

        return sm, stq


    def ensemble(xtrain, ytrain, xtest, epochs3, ens):
        # xtrain = x_train
        # ytrain = y_train
        # xtest = x_test
        # epochs3 = 1
        # ens = 5

        from tensorflow.keras.losses import categorical_crossentropy
        print('ENSEMBLE')
        mcpr = []
        nop = 10

        if ens_tr == 0:
            for t in range(ens):
                print('MODEL: ', t)
                model7 = tf.keras.Sequential()
                model7.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), input_shape=xtrain[0].shape, activation='tanh',
                                  padding="same"))  # , activation='tanh'
                model7.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
                model7.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                  padding='valid'))  # , activation='tanh'
                model7.add(Flatten())
                model7.add(Dense(120, activation='tanh'))
                model7.add(Dense(84, activation='tanh'))
                model7.add(Dense(10, activation = tf.nn.softmax))
                model7.load_weights(os.path.join(ens1, str(t) + '.h5'))

                mcpr.append(model7.predict(xtest))
        else:
            for i in range(ens):
                print('MODEL: ', i)
                model7 = tf.keras.Sequential()
                model7.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), input_shape=xtrain[0].shape, activation='tanh',
                                  padding="same"))  # , activation='tanh'
                model7.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
                model7.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                  padding='valid'))  # , activation='tanh'
                model7.add(Flatten())
                model7.add(Dense(120, activation='tanh'))
                model7.add(Dense(84, activation='tanh'))
                model7.add(Dense(10, activation = tf.nn.softmax))

                # model6.summary()

                model7.compile(loss= categorical_crossentropy, optimizer=tf.optimizers.Adam(),
                               metrics=['accuracy'])
                model7.fit(xtrain, ytrain, epochs=epochs3, verbose=2)

                ens_pt = os.path.join(ens1, str(i) + '.h5')
                model7.save(ens_pt)

                mcpr.append(model7.predict(xtest))

        mcarr = np.asarray(mcpr)
        mcpr_m = np.mean(mcarr, 0)
        mcpr_s = np.std(mcpr, 0)

        pr11 = mcpr_m
        pr1 = np.zeros((np.shape(mcpr_m)[0], np.shape(mcpr_m)[1]))
        aa1 = np.max(pr11, 1)
        for i in range(len(aa1)):
            for j in range(nop):
                if pr11[i, j] == aa1[i]:
                    pr1[i, j] = 1
                else:
                    pr1[i, j] = 0

        sm = np.sum(mcpr_s*pr1, 1)

        return sm, mcpr_s

    def svi_ll(model_1, xtrain, xtest, ytrain, epochs):
        import tensorflow_probability as tfp
        from tensorflow_probability.python.layers.dense_variational import DenseFlipout
        from tensorflow_probability.python.layers import dense_variational
        from tensorflow_probability.python.layers import conv_variational

        if mdtr == 0:
            model_1.load_weights(pt)

        for layer in model_1.layers[:-1]:
            layer.trainable = True

        def get_neg_log_likelihood_fn(bayesian=False):
            """
            Get the negative log-likelihood function
            # Arguments
                bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

            # Returns
                a negative log-likelihood function
            """
            if bayesian:
                def neg_log_likelihood_bayesian(y_true, y_pred):
                    labels_distribution = tfp.distributions.Categorical(logits=y_pred)
                    log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
                    loss = -tf.reduce_mean(input_tensor=log_likelihood)
                    return loss

                return neg_log_likelihood_bayesian
            else:
                def neg_log_likelihood(y_true, y_pred):
                    y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
                    loss = keras.losses.categorical_crossentropy(y_true, y_pred_softmax)
                    return loss

                return neg_log_likelihood

        fc = []
        for i in range(len(model_1.layers)):
            fc.append(model_1.layers[i])

        model5 = tf.keras.Sequential()
        for i in range(len(model_1.layers)-2):
            model5.add(model_1.layers[i])

        model5.add(DenseFlipout(10))

        if mdtr == 0 and svlltr == 0:
            model5.load_weights(svill_pt)

        model5.summary()
        print('SVI_LL')

        if svlltr == 1:
            model5.compile(loss=get_neg_log_likelihood_fn(bayesian=True), optimizer=tf.optimizers.Adam(1e-3), metrics=['accuracy'])
            model5.fit(xtrain, ytrain, batch_size=10000, epochs=epochs, verbose=2)
            model5.save(svill_pt)

        sm = model5.predict(xtest)

        pr1 = np.zeros((np.shape(sm)[0], np.shape(sm)[1]))
        aa1 = np.max(sm, 1)
        for i in range(len(aa1)):
            for j in range(10):
                if sm[i, j] == aa1[i]:
                    pr1[i, j] = 1
                else:
                    pr1[i, j] = 0

        mm = np.sum(sm*pr1, 1)

        return mm, sm

    def svi(model_1, xtrain, xtest, ytrain, epochs):
        import tensorflow_probability as tfp
        from tensorflow_probability.python.layers.dense_variational import DenseFlipout
        from tensorflow_probability.python.layers import dense_variational
        from tensorflow_probability.python.layers import Convolution2DFlipout

        if mdtr == 0:
            model_1.load_weights(pt)

        def get_neg_log_likelihood_fn(bayesian=False):
            """
            Get the negative log-likelihood function
            # Arguments
                bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

            # Returns
                a negative log-likelihood function
            """
            if bayesian:
                def neg_log_likelihood_bayesian(y_true, y_pred):
                    labels_distribution = tfp.distributions.Categorical(logits=y_pred)
                    log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
                    loss = -tf.reduce_mean(input_tensor=log_likelihood)
                    return loss

                return neg_log_likelihood_bayesian
            else:
                def neg_log_likelihood(y_true, y_pred):
                    y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
                    loss = keras.losses.categorical_crossentropy(y_true, y_pred_softmax)
                    return loss

                return neg_log_likelihood

        model6 = tf.keras.Sequential()
        model6.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), input_shape=xtrain[0].shape,activation='tanh', padding="same")) #, activation='tanh'
        model6.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model6.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')) #, activation='tanh'
        model6.add(Flatten())
        model6.add(DenseFlipout(120, activation='tanh'))
        model6.add(DenseFlipout(84, activation='tanh'))
        model6.add(DenseFlipout(10))

        if mdtr == 0 and svtr == 0:
            model6.load_weights(svi_pt)

        model6.summary()
        print('SVI')

        if svtr == 1:
            model6.compile(loss=get_neg_log_likelihood_fn(bayesian=True), optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
            model6.fit(xtrain, ytrain,batch_size=len(xtrain), epochs=epochs, verbose=2)
            model6.save(svi_pt)

        sm = model6.predict(xtest)

        pr1 = np.zeros((np.shape(sm)[0], np.shape(sm)[1]))
        aa1 = np.max(sm, 1)
        for i in range(len(aa1)):
            for j in range(10):
                if sm[i, j] == aa1[i]:
                    pr1[i, j] = 1
                else:
                    pr1[i, j] = 0

        mm = np.sum(sm*pr1, 1)

        return mm, sm

