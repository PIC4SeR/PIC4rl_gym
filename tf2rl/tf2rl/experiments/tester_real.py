import os
import time
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if tf.config.experimental.list_physical_devices("GPU"):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class TesterReal:
    """
    Tester class for off-policy reinforcement learning on real robots

    Command Line Args:

        * ``--max-steps`` (int): The maximum steps for training. The default is ``int(1e6)``
        * ``--episode-max-steps`` (int): The maximum steps for an episode. The default is ``int(1e3)``
        * ``--n-experiments`` (int): Number of experiments. The default is ``1``
        * ``--show-progress``: Call ``render`` function during training
        * ``--save-model-interval`` (int): Interval to save model. The default is ``int(1e4)``
        * ``--save-summary-interval`` (int): Interval to save summary. The default is ``int(1e3)``
        * ``--model-dir`` (str): Directory to restore model.
        * ``--dir-suffix`` (str): Suffix for directory that stores results.
        * ``--normalize-obs``: Whether normalize observation
        * ``--logdir`` (str): Output directory name. The default is ``"results"``
        * ``--evaluate``: Whether evaluate trained model
        * ``--test-interval`` (int): Interval to evaluate trained model. The default is ``int(1e4)``
        * ``--show-test-progress``: Call ``render`` function during evaluation.
        * ``--test-episodes`` (int): Number of episodes at test. The default is ``5``
        * ``--save-test-path``: Save trajectories of evaluation.
        * ``--show-test-images``: Show input images to neural networks when an episode finishes
        * ``--save-test-movie``: Save rendering results.
        * ``--use-prioritized-rb``: Use prioritized experience replay
        * ``--use-nstep-rb``: Use Nstep experience replay
        * ``--n-step`` (int): Number of steps for nstep experience reward. The default is ``4``
        * ``--logging-level`` (DEBUG, INFO, WARNING): Choose logging level. The default is ``INFO``
    """

    def __init__(self, policy, env, args, test_env=None):
        """
        Initialize Tester class

        Args:
            policy: Policy to be trained
            env (gym.Env): Environment for train
            args (Namespace or dict): config parameters specified with command line
            test_env (gym.Env): Environment for test.
        """
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(TesterReal.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(
                shape=env.observation_space.shape
            )

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args,
            user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix),
        )
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir,
        )

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5
        )

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def __call__(self):
        """
        Execute testing
        """
        if self._evaluate:
            self.evaluate_policy_continuously()

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error(
                "Please specify model directory by passing command line argument `--model-dir`"
            )
            exit(-1)
        i = 0
        self.evaluate_policy(i)
        while True:
            i = i + 1
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))

            self.evaluate_policy(i)

    def evaluate_policy(self, episode=None):
        done = False
        obs = self._test_env.reset(episode)  # use the reset to put the robot in pause
        while not done:
            action = self._policy.get_action(obs, test=True)
            next_obs, _, done, _ = self._test_env.step(action)
            obs = next_obs
        return

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (
            args.episode_max_steps
            if args.episode_max_steps is not None
            else args.max_steps
        )
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir

        # test settings
        self._evaluate = args.evaluate
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler="resolve")
        # experiment settings
        parser.add_argument(
            "--max-steps",
            type=int,
            default=int(1e6),
            help="Maximum number steps to interact with env.",
        )
        parser.add_argument(
            "--episode-max-steps",
            type=int,
            default=int(1e3),
            help="Maximum steps in an episode",
        )
        parser.add_argument(
            "--n-experiments", type=int, default=1, help="Number of experiments"
        )
        parser.add_argument(
            "--show-progress",
            action="store_true",
            help="Call `render` in training process",
        )
        parser.add_argument(
            "--save-model-interval",
            type=int,
            default=int(1e4),
            help="Interval to save model",
        )
        parser.add_argument(
            "--save-summary-interval",
            type=int,
            default=int(1e3),
            help="Interval to save summary",
        )
        parser.add_argument(
            "--model-dir", type=str, default=None, help="Directory to restore model"
        )
        parser.add_argument(
            "--dir-suffix",
            type=str,
            default="",
            help="Suffix for directory that contains results",
        )
        parser.add_argument(
            "--normalize-obs", action="store_true", help="Normalize observation"
        )
        parser.add_argument(
            "--logdir", type=str, default="results", help="Output directory"
        )
        parser.add_argument(
            "--policy", type=str, default="DQN", help="Policy used for training"
        )
        parser.add_argument(
            "--policy_trainer",
            type=str,
            default="off-policy",
            help="Policy type, on-policy or off-policy",
        )
        parser.add_argument(
            "--change_goal_and_pose",
            type=int,
            default="20",
            help="How many step for each goal-pose pair",
        )
        parser.add_argument(
            "--starting_episodes",
            type=int,
            default="400",
            help="How many episodes with random goals",
        )
        parser.add_argument(
            "--tflite_flag", type=bool, default="False", help="Use or not tflite models"
        )
        parser.add_argument(
            "--tflite_model_path",
            type=str,
            default="~/inference/actor_fp16.tflite",
            help="Path of tflite model",
        )
        # test settings
        parser.add_argument(
            "--evaluate", action="store_true", help="Evaluate trained model"
        )
        parser.add_argument(
            "--test-interval",
            type=int,
            default=int(1e4),
            help="Interval to evaluate trained model",
        )
        parser.add_argument(
            "--show-test-progress",
            action="store_true",
            help="Call `render` in evaluation process",
        )
        parser.add_argument(
            "--test-episodes",
            type=int,
            default=5,
            help="Number of episodes to evaluate at once",
        )
        parser.add_argument(
            "--save-test-path",
            action="store_true",
            help="Save trajectories of evaluation",
        )
        parser.add_argument(
            "--show-test-images",
            action="store_true",
            help="Show input images to neural networks when an episode finishes",
        )
        parser.add_argument(
            "--save-test-movie", action="store_true", help="Save rendering results"
        )
        # replay buffer
        parser.add_argument(
            "--use-prioritized-rb",
            action="store_true",
            help="Flag to use prioritized experience replay",
        )
        parser.add_argument(
            "--use-nstep-rb",
            action="store_true",
            help="Flag to use nstep experience replay",
        )
        parser.add_argument(
            "--n-step", type=int, default=4, help="Number of steps to look over"
        )
        # others
        parser.add_argument(
            "--logging-level",
            choices=["DEBUG", "INFO", "WARNING"],
            default="INFO",
            help="Logging level",
        )
        return parser
