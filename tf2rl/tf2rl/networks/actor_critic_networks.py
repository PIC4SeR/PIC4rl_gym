#!/usr/bin/env python3

import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal, HeUniform, GlorotUniform
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
import numpy as np

class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action=(0.4,1.), min_action=(0.,-1.), units=(256, 256), name="Actor"):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_dim = action_dim
        #self.l1 = Dense(units[0], name="L1")
        #self.l2 = Dense(units[1], name="L2")
        #self.l3 = Dense(units[2], name="L3")

        self.base_layers = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers.append(Dense(unit, activation='relu'))

        #Output Layer
        self.v_out = Dense(1, activation = 'sigmoid')
        self.w_out = Dense(1, activation = 'tanh')

        self.max_action = tf.cast(max_action, dtype = tf.float32)
        self.min_action = tf.cast(min_action, dtype = tf.float32)

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, inputs):
        #features = tf.nn.relu(self.l1(inputs))
        #features = tf.nn.relu(self.l2(features))
        #features = tf.nn.relu(self.l3(features))
        features = inputs
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        
        Linear_velocity = self.v_out(features)*self.max_action[0]
        Angular_velocity = self.w_out(features)*self.max_action[1]
        action = concatenate([Linear_velocity, Angular_velocity])
        return action

    def model(self):
        return tf.keras.Model(inputs = self.state_input, outputs = self.call(self.state_input), name = self.model_name)
    
class ActorTanh(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, min_action, units=(256, 256), name="Actor"):
        super().__init__(name=name)
        
        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_dim = action_dim

        # Base Layers
        self.base_layers = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers.append(Dense(unit, activation='relu'))

        # Output Layer
        self.out_layer = Dense(action_dim, activation = 'tanh')

        self.max_action = tf.cast(max_action, dtype = tf.float32)
        self.min_action = tf.cast(min_action, dtype = tf.float32)
        self.act_width = (self.max_action - self.min_action)*0.5
        self.act_bias = (self.max_action + self.min_action)*0.5
        
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, features):
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        action = self.out_layer(features)
        action = tf.multiply(action,self.act_width)
        action += self.act_bias
        return action

    def model(self):
        return tf.keras.Model(inputs = self.state_input, outputs = self.call(self.state_input), name = self.model_name)

class ConvActor(tf.keras.Model):
    def __init__(self, state_shape, image_shape=(112,112,1,), action_dim=2, max_action=(0.4,1.), min_action=(0.,-1.), units=(256, 256),
        num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name="Actor"):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_dim = action_dim

        self.image_shape = image_shape
        self.state_info_shape = state_shape[0]-(self.image_shape[0]*self.image_shape[1])

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers.append(Dense(unit, activation='relu'))

        #Output Layer
        self.v_out = Dense(1, activation = 'sigmoid')
        self.w_out = Dense(1, activation = 'tanh')

        self.max_action = tf.cast(max_action, dtype = tf.float32)
        self.min_action = tf.cast(min_action, dtype = tf.float32)

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, states):
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        features = tf.concat((state_info,features), axis=1)

        for cur_layer in self.base_layers:
            features = cur_layer(features)
        
        Linear_velocity = self.v_out(features)*self.max_action[0]
        Angular_velocity = self.w_out(features)*self.max_action[1]
        action = concatenate([Linear_velocity, Angular_velocity])
        return action

    def model(self):
        return tf.keras.Model(inputs = self.state_input, outputs = self.call(self.state_input), name = self.model_name)

class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(256, 256), name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        #self.l3 = Dense(units[2], name="L3")
        self.lout = Dense(1, name="Lout")

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        #features = tf.nn.relu(self.l3(features))
        values = self.lout(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf'):
        super().__init__(name=name)
	
        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))
        self.base_layers = []
        for i in range(len(critic_units)):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class CriticTD3(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))
        self.l3 = None
        self.l6 = None

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        if len(units) > 2:
            self.l3 = Dense(units[2], name="L3")
        self.q1 = Dense(1, name="Q1")

        self.l4 = Dense(units[0], name="L4")
        self.l5 = Dense(units[1], name="L5")
        if len(units) > 2:
            self.l6 = Dense(units[2], name="L6")
        self.q2 = Dense(1, name="Q2")

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1, action_dim), dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)

    def call(self, states, actions):
        xu = tf.concat((states, actions), axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        if self.l3 != None:
            x1 = tf.nn.relu(self.l3(x1))
        x1 = self.q1(x1)

        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        if self.l6 != None:
            x2 = tf.nn.relu(self.l6(x2))
        x2 = self.q2(x2)

        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)

    def Q1(self, states, actions):
        xu = tf.concat((states, actions), axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        if self.l3 != None:
            x1 = tf.nn.relu(self.l3(x1))
        x1 = self.q1(x1)

        return tf.squeeze(x1, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class CriticTD3_v2(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.base_layers1 = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers1.append(Dense(unit, activation='relu'))

        self.out_layer1 = Dense(1, name="Q1", activation='linear')

        self.base_layers2 = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers2.append(Dense(unit, activation='relu'))
        self.out_layer2 = Dense(1, name="Q2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1, action_dim), dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)

    def call(self, states, actions):
        xu = tf.concat((states, actions), axis=1)
        x1 = self.base_layers1[0](xu)
        for cur_layer1 in self.base_layers1[1:]:
            x1 = cur_layer1(x1)
        x1 = self.out_layer1(x1)

        x2 = self.base_layers2[0](xu)
        for cur_layer2 in self.base_layers2[1:]:
            x2 = cur_layer2(x1)
        x2 = self.out_layer2(x1)

        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)

    def Q1(self, states, actions):
        x1 = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers1:
            x1 = cur_layer(x1)
        x1 = self.out_layer1(x1)
        return tf.squeeze(x1, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class ConvCriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), 
                 num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name='qf'):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[1], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(len(critic_units)):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        features = tf.concat((features, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class ConvMixCriticQ(tf.keras.Model):
    def __init__(self, state_shape, image_shape=(112,112,1,), action_dim=2, critic_units=(256, 256), 
                 num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name='qf'):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.image_shape = image_shape
        self.state_info_shape = state_shape[0] - (self.image_shape[0]*self.image_shape[1])

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(len(critic_units)):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        features = tf.concat((state_info,features,actions), axis=1)

        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class ConvMixCriticTD3(tf.keras.Model):
    def __init__(self, state_shape, image_shape=(112,112,1,), action_dim=2, units=(256, 256), 
                 num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name='qf'):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.image_shape = image_shape
        self.image_shape = (112,112,1,)
        self.state_info_shape = state_shape[0] -(self.image_shape[0]*self.image_shape[1])

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.l3 = None
        self.l6 = None

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        if len(units) > 2:
            self.l3 = Dense(units[2], name="L3")
        self.q1 = Dense(1, name="Q1")

        self.l4 = Dense(units[0], name="L4")
        self.l5 = Dense(units[1], name="L5")
        if len(units) > 2:
            self.l6 = Dense(units[2], name="L6")
        self.q2 = Dense(1, name="Q2")

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        xu = tf.concat((state_info,features,actions), axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        if self.l3 != None:
            x1 = tf.nn.relu(self.l3(x1))
        x1 = self.q1(x1)

        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        if self.l6 != None:
            x2 = tf.nn.relu(self.l6(x2))
        x2 = self.q2(x2)

        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)

    def Q1(self, states, actions):
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        xu = tf.concat((state_info,features,actions), axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        if self.l3 != None:
            x1 = tf.nn.relu(self.l3(x1))
        x1 = self.q1(x1)
        return tf.squeeze(x1, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)