import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.layers as L
import _pickle as pickle
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence
from keras.utils.data_utils import Sequence
from keras.regularizers import l2
from keras.constraints import non_neg, Constraint
from keras_exp.multigpu import get_available_gpus, make_parallel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from kera

def model_create(ARGS):
    def DxCM(ARGS):
        reshape_size = ARGS.emb_size

        beta_activation = 'relu'

        def reshape(data):
            """Reshape the context vectors to 3D vector"""
            return K.reshape(x=data, shape=(K.shape(data)[0], 1, reshape_size))

        #Code Input
        diag = L.Input((None, None), name='diag_input')
        proc = L.Input((None, None), name='proc_input')
        time = L.Input((None, None), name='time_input')
        #claim_sup = L.Input((None, None), name='claim_sup')

        inputs_list = [diag, proc, time]
        #Calculate embedding for each code and sum them to a visit level
        diag_embs_total = L.Embedding(ARGS.num_diag+1,
                                      ARGS.emb_size,
                                      name='diag_embedding',
                                      weights = [np.load(ARGS.pretrained_diag_embedding,allow_pickle=True)],
                                      trainable=False,
                                      #embeddings_constraint=embeddings_constraint
                                      )(diag)

        proc_embs_total = L.Embedding(ARGS.num_proc+1,
                                      ARGS.emb_size,
                                      name='proc_embedding')(proc) # no constraint
        
        time_embs_total = L.Embedding(ARGS.num_time+1,
                                      ARGS.emb_size,
                                      name='time_embedding')(time) # no constraint
                                      
        diag_embs = L.Lambda(lambda x: K.sum(x, axis=2))(diag_embs_total)
        proc_embs = L.Lambda(lambda x: K.sum(x, axis=2))(proc_embs_total)
        time_embs = L.Lambda(lambda x: K.sum(x, axis=2))(time_embs_total)

        diag_embs = L.Dropout(ARGS.dropout_input)(diag_embs)
        proc_embs = L.Dropout(ARGS.dropout_input)(proc_embs)
        time_embs = L.Dropout(ARGS.dropout_input)(time_embs)

        full_embs = L.concatenate([diag_embs, proc_embs, time_embs], name='full_embs')
        alpha = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True), name='alpha')
        beta = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True), name='beta')

        alpha_dense = L.Dense(1, kernel_regularizer=l2(ARGS.l2))
        beta_dense = L.Dense(ARGS.emb_size+ARGS.numeric_size, activation=beta_activation, kernel_regularizer=l2(ARGS.l2))

        alpha_out = alpha(full_embs)
        alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
        alpha_out = L.Softmax(axis=1)(alpha_out)
        beta_out = beta(full_embs)
        beta_out = L.TimeDistributed(beta_dense, name='beta_dense_0')(beta_out)
        c_t = L.Multiply()([alpha_out, beta_out, diag_embs])
        c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)
        contexts = L.Lambda(reshape)(c_t)
        contexts = L.Dropout(ARGS.dropout_context)(contexts)
        output_layer = L.Dense(1, activation=None, name='dOut', kernel_initializer= initializers.RandomUniform(0, 1000), kernel_regularizer=l2(ARGS.l2))
        output = L.TimeDistributed(output_layer, name='time_distributed_out')(contexts)
        model = Model(inputs=inputs_list, outputs=[output])
        return model

    K.clear_session()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.Session(config=config)
    K.set_session(tfsess)
    model_final = DxCM(ARGS)

    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    model_final.compile(optimizer='adamax', loss=tf.losses.huber_loss(delta=30000.0), metrics=['mae', 'mse', coeff_determination], sample_weight_mode=None)
    #model_final.compile(optimizer='adamax', loss='mean_squared_error', metrics=['mae', 'mse', coeff_determination], sample_weight_mode=None)

    return model_final
