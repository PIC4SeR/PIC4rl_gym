import unittest
import numpy as np

from tf2rl.policies.tfp_categorical_actor import CategoricalActor
from tests.policies.common import CommonModel


class TestCategoricalActor(CommonModel):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = CategoricalActor(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            units=[4, 4])

    def test_call(self):
        # Single input
        state = np.random.rand(
            1, self.discrete_env.observation_space.low.size).astype(np.float32)
        self._test_call(
            inputs=state,
            expected_action_shapes=(1,),
            expected_log_prob_shapes=(1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size).astype(np.float32)
        self._test_call(
            inputs=states,
            expected_action_shapes=(self.batch_size,),
            expected_log_prob_shapes=(self.batch_size,))

    def test_compute_log_probs(self):
        # Single input
        state = np.random.rand(
            1, self.discrete_env.observation_space.low.size).astype(np.float32)
        action = np.random.randint(
            self.discrete_env.action_space.n, size=1)
        self._test_compute_log_probs(
            states=state,
            actions=action,
            expected_shapes=(1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size).astype(np.float32)
        actions = np.random.randint(
            self.discrete_env.action_space.n, size=self.batch_size)
        self._test_compute_log_probs(
            states=states,
            actions=actions,
            expected_shapes=(self.batch_size,))

    def test_compute_prob(self):
        # Single input
        state = np.random.rand(
            1, self.discrete_env.observation_space.low.size).astype(np.float32)
        result = self.policy.compute_prob(state)
        expected_shape = (1, self.discrete_env.action_space.n)
        self.assertEqual(result.shape, expected_shape)
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size).astype(np.float32)
        results = self.policy.compute_prob(states)
        expected_shape = (self.batch_size, self.discrete_env.action_space.n)
        self.assertEqual(results.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
