"""PPO training script for the drone racing environment using Brax."""

import functools
import sys
from pathlib import Path
from typing import Any

from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

# Add env directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "env"))
from env import Env

from mujoco_playground._src.wrapper import wrap_for_brax_training


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def progress_fn(step: int, metrics: dict[str, Any]) -> None:
    """Log training progress."""
    reward_key = None
    for key in ["eval/episode_reward", "eval/episode_reward_mean"]:
        if key in metrics:
            reward_key = key
            break

    if reward_key:
        logging.info(
            "step=%d, reward=%.2f",
            step,
            metrics[reward_key],
        )
    else:
        logging.info("step=%d", step)


def train(
    num_timesteps: int = 10_000_000,
    episode_length: int = 1000,
    save_checkpoint_path: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Train a PPO policy for drone racing.

    Args:
        num_timesteps: Total number of environment steps for training.
        episode_length: Maximum steps per episode.
        save_checkpoint_path: Path to save model checkpoints. Defaults to
            policy/checkpoints/.

    Returns:
        Tuple of (make_policy function, network params, training metrics).
    """
    logging.set_verbosity(logging.INFO)

    # Set default checkpoint path.
    if save_checkpoint_path is None:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        save_checkpoint_path = str(CHECKPOINT_DIR)

    env = Env()
    wrapped_env = wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=1,
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        network_factory=network_factory,
        save_checkpoint_path=save_checkpoint_path,
        wrap_env=False,
    )

    make_policy, params, metrics = train_fn(
        environment=wrapped_env,
        progress_fn=progress_fn,
    )

    return make_policy, params, metrics


if __name__ == "__main__":
    make_policy, params, metrics = train()
    logging.info("Training complete. Final metrics: %s", metrics)
