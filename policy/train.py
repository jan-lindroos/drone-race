import functools
import os
import sys
from datetime import datetime
from pathlib import Path

# MJX only supports CUDA or CPU, not Metal.
os.environ["JAX_PLATFORMS"] = "cpu"

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from brax.training.agents.ppo import train as ppo

from env.env import DroneRace


def train():
    env = DroneRace()

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    train = functools.partial(
        ppo.train,
        num_timesteps=10_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=5000,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=24,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=3072,
        batch_size=512,
        seed=0,
        log_training_metrics=True,
        training_metrics_steps=20_000,
        save_checkpoint_path=str(checkpoint_dir),
    )

    def progress_callback(num_steps, metrics):
        msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} steps: {num_steps}"
        if "eval/episode_reward" in metrics:
            msg += f" reward: {metrics['eval/episode_reward']:.2f}"
        if "eval/episode_reward_std" in metrics:
            msg += f" (±{metrics['eval/episode_reward_std']:.2f})"
        print(msg)

    make_inference_fn, params, _ = train(environment=env, progress_fn=progress_callback)

    return make_inference_fn, params

if __name__ == "__main__":
    train()
