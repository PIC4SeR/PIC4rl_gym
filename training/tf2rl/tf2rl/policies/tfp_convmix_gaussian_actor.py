import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import RandomUniform, glorot_normal, HeUniform, GlorotUniform
layers = tf.keras.layers


class ConvGaussianActor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, min_action, num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3),
                 units = (256,256), hidden_activation="relu",
                 state_independent_std=False,
                 squash=False, name='gaussian_policy'):
        super().__init__(name=name)

        self.model_name = name
        self.action_dim = action_dim
        self._state_independent_std = state_independent_std
        self._squash = squash
        self.image_shape = (112,112,1,)
        self.state_info_shape=2
        self.state_shape=state_shape
 
        self.state_input = Input(shape=self.state_shape)

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers.append(layers.Dense(unit, activation=hidden_activation))

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action
        self._min_action = min_action

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + self.state_shape, dtype=np.float32))

        self(dummy_state)

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        features = tf.concat([state_info, features], -1)

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states)
        if test:
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        #actions = tf.multiply(actions,tf.constant([self._max_action[0]*0.5, self._max_action[1]], dtype=tf.float32))
        #actions += tf.constant([self._max_action[0]*0.5, 0.], dtype=tf.float32)
        act_width = (self._max_action - self._min_action)*0.5
        act_bias = (self._max_action + self._min_action)*0.5
        actions = tf.multiply(actions,act_width)
        actions += act_bias
        return actions, log_pis

    def compute_log_probs(self, states, actions):
        raw_actions = actions / self._max_action
        dist = self._compute_dist(states)
        logp_pis = dist.log_prob(raw_actions)
        return logp_pis

    def compute_entropy(self, states):
        dist = self._compute_dist(states)
        return dist.entropy()

    def model(self):
        return tf.keras.Model(inputs = self.state_input, outputs = self.call(self.state_input), name = self.model_name)
