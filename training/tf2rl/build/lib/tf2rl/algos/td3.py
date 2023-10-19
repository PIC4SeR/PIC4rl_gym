import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from tf2rl.algos.ddpg import DDPG
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.networks.actor_critic_networks import Actor, ConvActor, CriticTD3, ConvMixCriticTD3


class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic policy gradient (TD3) Agent: https://arxiv.org/abs/1802.09477

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size for training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e6``.
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            image_shape=(112,112,1),
            name="TD3",
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            max_action=(0.5,1.),
            min_action=(0.,-1.),
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=(256,128,128),
            critic_units=(256,256),
            network='mlp',
            sigma=0.15,
            tau=0.005,
            epsilon = 1.0, 
            epsilon_decay = 0.998, 
            epsilon_min = 0.05,
            log_level = 20,
            **kwargs):
        """
        Initialize TD3

        Args:
            shate_shape (iterable of ints): Observation state shape
            action_dim (int): Action dimension
            name (str): Network name. The default is ``"TD3"``.
            actor_update_freq (int): Number of critic updates per one actor upate.
            policy_noise (float):
            noise_clip (float):
            critic_units (iterable of int): Numbers of units at hidden layer of critic. The default is ``(400, 300)``
            max_action (float): Size of maximum action. (``-max_action`` <= action <= ``max_action``). The degault is ``1``.
            lr_actor (float): Learning rate for actor network. The default is ``0.001``.
            lr_critic (float): Learning rage for critic network. The default is ``0.001``.
            actor_units (iterable of int): Number of units at hidden layers of actor.
            sigma (float): Standard deviation of Gaussian noise. The default is ``0.1``.
            tau (float): Weight update ratio for target network. ``target = (1-tau)*target + tau*network`` The default is ``0.005``.
            n_warmup (int): Number of warmup steps before training. The default is ``1e4``.
            memory_capacity (int): Replay Buffer size. The default is ``1e4``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(name=name, state_shape=state_shape, action_dim=action_dim, **kwargs)

        # Define and initialize Actor network
        if network=='mlp':
            self.actor = Actor(state_shape, action_dim, max_action, min_action, actor_units)
            self.actor_target = Actor(state_shape, action_dim, max_action, min_action, actor_units)
        elif network=='conv':
            self.actor = ConvActor(state_shape, image_shape, action_dim, max_action, min_action, actor_units)
            self.actor_target = ConvActor(state_shape, image_shape, action_dim, max_action, min_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)
        if log_level < 20:
            self.actor.model().summary()

        if network=='mlp':
            self.critic = CriticTD3(state_shape, action_dim, critic_units, name="Q1_Q2")
            self.critic_target = CriticTD3(state_shape, action_dim, critic_units, name="target_Q1_Q2")
        elif network=='conv':
            self.critic = ConvMixCriticTD3(state_shape, image_shape, action_dim, critic_units, name="Q1_Q2")
            self.critic_target = ConvMixCriticTD3(state_shape, image_shape, action_dim, critic_units, name="target_Q1_Q2")
        self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_critic)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)
        if log_level < 20:
            self.critic.model().summary()

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        self._it = tf.Variable(0, dtype=tf.int32)
        
        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min


    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors1, td_errors2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                critic_loss = (tf.reduce_mean(td_errors1 ** 2 * weights) +
                               tf.reduce_mean(td_errors2 ** 2 * weights))

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            self._it.assign_add(1)
            with tf.GradientTape() as tape:
                sample_actions = self.actor(states)
                actor_loss = - tf.reduce_mean(self.critic.Q1(states, sample_actions))

            remainder = tf.math.mod(self._it, self._actor_update_freq)

            def optimize_actor():
                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                return self.actor_optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables))

            tf.cond(pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, tf.abs(td_errors1) + tf.abs(td_errors2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """
        Compute TD error

        Args:
            states
            actions
            next_states
            rewars
            dones

        Returns:
            np.ndarray: Sum of two TD errors.
        """
        td_errors1, td_errors2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_errors1.numpy()) + np.abs(td_errors2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        assert len(dones.shape) == 2
        assert len(rewards.shape) == 2
        rewards = tf.squeeze(rewards, axis=1)
        dones = tf.squeeze(dones, axis=1)

        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action),
                                 stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float32)
            next_action = tf.clip_by_value(
                next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)

            next_q1_target, next_q2_target = self.critic_target(next_states, next_action)
            next_q_target = tf.minimum(next_q1_target, next_q2_target)
            q_target = tf.stop_gradient(rewards + not_dones * self.discount * next_q_target)
            current_q1, current_q2 = self.critic(states, actions)

        return q_target - current_q1, q_target - current_q2
