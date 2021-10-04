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

        # x = fc[2].output
        # for g in range(3, len(model.layers)-4):
        #     x = fc[g](x)
        #     # x = Dropout(dpr)(x)
        #
        # x = fc[-4](x)
        # x = PermaDropout(dpr)(x)
        # x = fc[-2](x)
        # predictors = fc[-1](x)

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

    def qipf(model, xtest, test_labels, bwf, xtr, orr, ttt):

        ########### IMPORT NECESSARY PYTHON LIBRARIES ##########
        import math
        import skimage.measure as sk
        from statsmodels.nonparametric.bandwidths import bw_silverman as bw

        ########## LOAD SAVED MODEL WEIGHTS IN LOCATION pt ####################
        if mdtr == 0:
            model.load_weights(pt)

        ####################### EXTRACT MODEL LAYERS ################################
        fc = []
        for i in range(len(model.layers)):
            fc.append(model.layers[i])


        ########## OBTAIN REDUCED DIMENSION (BY AVERAGE POOLING) OF MODEL WEIGHTS ############
        hm1 = sk.block_reduce(fc[0].get_weights()[0], (1, 5, 1, 1), np.mean).flatten()
        hm2 = sk.block_reduce(fc[2].get_weights()[0], (1, 5, 6, 1), np.mean).flatten()
        hm3 = sk.block_reduce(fc[5].get_weights()[0], (10, 10), np.mean).flatten()
        hm4 = sk.block_reduce(fc[6].get_weights()[0], (5, 8), np.mean).flatten()
        hm5 = sk.block_reduce(fc[7].get_weights()[0], (1, 5), np.mean).flatten()

        ###### COMBINE WEIGHTS OF ALL LAYERS #######
        hmt = np.concatenate((hm1, hm2, hm3, hm4, hm5))
        hmt = hmt.flatten()

        ####### NORMALIZE WEIGHTS #######
        hmt = (hmt - np.mean(hmt)) / np.std(hmt)


        ################ FORM NEW MODEL WITHOUT SOFTMAX FUNCTION ################
        x = fc[-3].output
        preds = fc[-2](x)
        modelq = Model(inputs=model.input, outputs=preds)

        ######## GET TEST SET PRE-SOFTMAX OUTPUT #######
        pmt = modelq.predict(xtest)
        pmt1 = np.max(pmt, 1)

        ######## GET TRAIN SET PRE-SOFTMAX OUTPUT #######
        rmt = modelq.predict(xtr)
        rmt1 = np.max(rmt, 1)

        ####### NORMALIZE TEST OUTPUTS (WRT TO TRAIN OUTPUTS) #######
        pmtn = (pmt1 - np.mean(rmt1)) / np.std(rmt1)

        ############ NEIGHBORHOOD SIZE FOR GRADIENT (LAPLACIAN) COMPUTATION: TAKEN AS THE DATA STD. DEV #########
        es = 0.2

        ################## C-QIPF INITIALIZATIONS #####################
        s0m = []
        sqp = []
        w1 = np.zeros((len(xtest), 3))
        ct = 0

        ################## KERNEL WIDTH #####################
        sigg1 = bwf * np.average(bw(hmt))


        N1 = len(xtest)
        N2 = len(hmt)

        ################## START C-QIPF BETWEEN MODEL WEIGHTS AND PRE-SOFTMAX TEST OUTPUTS #####################
        print('start')
        start = time.time()
        for i in range(len(xtest)):

            ##### NEIGHBORHOOD AROUND CURRENT TEST OUTPUT TO MEASURE LAPLACIAN #########
            jh = pmtn[i]
            jh = [jh - es, jh, jh + es]

            ##### KEEP TRACK OF ITERATION NUMBER ######
            ct += 1
            if ct % 500 == 0:
                print(" iter #: ", ct, ' / ', N1)

            #### EVALUATE WAVE-FUNCTION AT A SAMPLE LOCATION AND NEIGHBORING POINTS ###################
            w1[i, 0] = (1 / N2) * np.sum(np.exp(-(np.power(jh[0] - hmt, 2)) / (2 * sigg1 ** 2)))
            w1[i, 1] = (1 / N2) * np.sum(np.exp(-(np.power(jh[1] - hmt, 2)) / (2 * sigg1 ** 2)))
            w1[i, 2] = (1 / N2) * np.sum(np.exp(-(np.power(jh[2] - hmt, 2)) / (2 * sigg1 ** 2)))

            jh = [[w1[i, 0]], [w1[i, 1]], [w1[i, 2]]]
            w0 = np.sqrt(jh)
            x = w0

            ##################### HERMITE DECOMPOSITION OF THE WAVE - FUNCTION #########################
            n = np.arange(1, orr + 1)
            fn = np.floor(n / 2)
            p = np.arange(0, orr + 1)
            x = 2 * x
            lex = len(x)
            lenn = len(n)

            if p[0] == 0:
                xp = np.power(x, p[1::])
                xp = np.concatenate([np.ones((lex, 1)), xp], axis=1)
            else:
                xp = np.power(x, p)

            H = np.zeros((lex, lenn))
            H = np.float64(H)
            yy = np.zeros(lenn)
            yy = np.float64(yy)

            for k in range(lenn):
                for m in range(int(fn[k]) + 1):
                    is_the_power = p == n[k] - (2 * m)
                    jj = (1 - 2 * np.mod(m, 2)) / math.factorial(m) / math.factorial(n[k] - (2 * m)) * xp[:,
                                                                                                       is_the_power]
                    H[:, k] += jj[:, 0] ###### HERMITE POLYNOMIAL PROJECTIONS

                ##### NORMALIZATION OF HERMITE PROJECTIONS ########
                ll = math.factorial(n[k])
                H[:, k] = ll * H[:, k]
            wy = H
            sg = sigg1 ** 2
            qe = np.gradient(np.gradient(np.abs(wy), axis=1), axis=1)
            qe1 = np.abs(wy)
            vc = np.multiply((sg / 2), np.divide(qe, qe1))
            r = np.zeros((np.shape(vc)[0], int((np.shape(vc)[1] / ttt)) - 1))
            for qk in range(1, int(orr / ttt)):
                if len(wy) == 1:
                    r[:, qk - 1] = np.abs(vc[:, (ttt * qk) - 1])
                else:
                    r[:, qk - 1] = vc[:, (ttt * qk) - 1] - np.min(vc[:, (ttt * qk) - 1]) ##### QIPF CORRESPONDING TO EACH MODE
            r = r.T
            qn0 = np.double(r[0:-1])
            sk = qn0
            qp = np.double(r[-1])

            s0m.append(sk)
            sqp.append(qp)

        stop = time.time()
        duration = stop - start
        print(duration)

        sm = s0m

        return sm, sqp, hmt, pmtn

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

