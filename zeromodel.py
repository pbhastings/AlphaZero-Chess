import constants
import chess
import chess.pgn
import btm

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Flatten,Reshape, add,Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import regularizers
#from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
import pydot

import pandas as pd
import sys
import numpy as np


#def build_zero_model(num_res_layers):
#    input = Input(shape=(17,8,8))
#    #initial branch
#    x = Conv2D(64, 3)(input)
#    x = BatchNormalization(axis=-1)(x)
#    x = LeakyReLU()(x)
#    x = Conv2D(64, 3)(x)
#    x = BatchNormalization(axis=-1)(x)
#    x = LeakyReLU()(x)

#    #policy branch
#    policy_branch = Conv2D(64, 3)(x)
#    policy_branch = BatchNormalization(axis=-1)(policy_branch)
#    policy_branch = LeakyReLU()(policy_branch)
#    policy_branch = Flatten()(policy_branch)
#    policy_branch = Dense(64*64)(policy_branch)

#    #value branch
#    value_branch = Conv2D(64, 3)(x)
#    value_branch = BatchNormalization(axis=-1)(value_branch)
#    value_branch = LeakyReLU()(value_branch)
#    value_branch = Flatten()(value_branch)
#    value_branch = Dense(128)(value_branch)
#    value_branch = LeakyReLU()(value_branch)
#    value_branch = Dense(1, activation='tanh')(value_branch) #we use tanh so that value is in [-1,1]

#    #build model
#    model = Model(inputs=input,outputs=[policy_branch,value_branch])
#    return model

def residual_layer(input_block, filters, kernel_size):

    x = conv_layer(input_block, filters, kernel_size)

    x = Conv2D(
    filters = filters
    , kernel_size = kernel_size
    , data_format="channels_first"
    , padding = 'same'
    , use_bias=False
    , activation='linear'
    , kernel_regularizer = regularizers.l2(0.0001)
    )(x)

    x = BatchNormalization(axis=1)(x)

    x = add([input_block, x])

    x = LeakyReLU()(x)

    return (x)

def conv_layer(x, filters, kernel_size):

    x = Conv2D(
    filters = filters
    , kernel_size = kernel_size
    , data_format="channels_first"
    , padding = 'same'
    , use_bias=False
    , activation='linear'
    , kernel_regularizer = regularizers.l2(0.0001)
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)

    return (x)

def value_head(x):

	x = Conv2D(
	filters = 1
	, kernel_size = (1,1)
	, data_format="channels_first"
	, padding = 'same'
	, use_bias=False
	, activation='linear'
	, kernel_regularizer = regularizers.l2(0.0001)
	)(x)


	x = BatchNormalization(axis=1)(x)
	x = LeakyReLU()(x)

	x = Flatten()(x)

	x = Dense(
		20
		, use_bias=False
		, activation='linear'
		, kernel_regularizer=regularizers.l2(0.0001)
		)(x)

	x = LeakyReLU()(x)

	x = Dense(
		1
		, use_bias=False
		, activation='tanh'
		, kernel_regularizer=regularizers.l2(0.0001)
		, name = 'value_head'
		)(x)



	return (x)

def policy_head(x):

	x = Conv2D(
	filters = 2
	, kernel_size = (1,1)
	, data_format="channels_first"
	, padding = 'same'
	, use_bias=False
	, activation='linear'
	, kernel_regularizer = regularizers.l2(0.0001)
	)(x)

	x = BatchNormalization(axis=1)(x)
	x = LeakyReLU()(x)

	x = Flatten()(x)

	x = Dense(
		64*64
		, use_bias=False
		, activation='linear'
		, kernel_regularizer=regularizers.l2(0.0001)
		, name = 'policy_head'
		)(x)

	return (x)

def policy_loss(y_true, y_pred):
    #we want to use the built in keras cross entropy from logits, so we replace illegal move values in p with -100
    illegal_moves = tf.equal(y_true, 0) #true where there is illegal move
    y_pred = tf.where(illegal_moves, [-100.], y_pred) #masking out illegal moves with -100
    return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

def build_zero_model(num_res_layers):
    num_layers = 12*constants.BOARD_MEMORY + 5
    input = Input(shape=(num_layers,8,8))
    x = conv_layer(input, 64, (3,3))
    #add res layers
    for i in range(num_res_layers):
        x = residual_layer(x, 64, (3,3))

    vh = value_head(x)
    ph = policy_head(x)

    losses = {'policy_head': policy_loss, 'value_head': 'mean_squared_error'}

    loss_weights = {
	    'policy_head': 0.5,
    	'value_head': 0.5
    }

    model = Model(inputs=[input], outputs=[ph, vh])

    model.compile(loss=losses, loss_weights=loss_weights, optimizer=SGD(learning_rate=0.1, momentum = 0.9))
    return model