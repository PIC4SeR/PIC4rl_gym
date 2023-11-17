import unittest

from tf2rl.algos.bi_res_ddpg import BiResDDPG
from tests.algos.common import CommonOffPolContinuousAlgos


class TestBiResDDPG(CommonOffPolContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = BiResDDPG(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
