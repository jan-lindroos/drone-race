from mujoco_playground._src.mjx_env import MjxEnv, State
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

from load_scene import load_scene


class Env(MjxEnv):
    """Drone racing environment with randomised gate courses."""

    def __init__(self, key: jax.Array):
        """Initialise environment with a random gate course.

        Args:
            key: JAX random key for course generation.
        """
        key1, key2 = jax.random.split(key)
        self.gate_count = jax.random.randint(key2, (), 4, 8)
        mj_model, self.gates = load_scene(
            key=key1,
            gate_count=self.gate_count,
            course_radius=self.gate_count * 2.0,
            vertical_deviation=2.0,
            horizontal_deviation=3.0,
        )

        # use recommended solver configuration, per
        # https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        self.drone_body_id = mj_model.body("drone").id

        super().__init__(mj_model)
        
    @property
    def action_size(self) -> int:
        return 4
    
    @property
    def observation_size(self) -> int:
        """Size of observation vector. 
        
        This includes the gyro (3) + accel (3) + quat (4) + vel (3) + target_dir (3) + next_dir (3) = 19.
        """
        return 19
        
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to initial state.

        Args:
            rng: JAX random key (unused, for API compatibility).

        Returns:
            Initial environment state.
        """
        data = mjx.make_data(self.mjx_model)
        data = mjx.forward(self.mjx_model, data)

        reward, done = jnp.zeros(2)
        current_gate = jnp.zeros(1)
        obs = self.get_obs(data, current_gate)
        info = {
            "current_gate": current_gate,
            "prev_pos": data.xpos[self.drone_body_id],
        }

        return State(data=data, obs=obs, reward=reward, done=done, metrics={}, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        """Advance the simulation by one timestep.

        Args:
            state: Current environment state.
            action: Motor thrust commands (4,).

        Returns:
            Updated environment state.
        """
        data = state.data.replace(ctrl=action)
        data = mjx.step(self.mjx_model, data)

        reward, new_gate, curr_pos = self.check_gate_passage(
            data, state.info["current_gate"], state.info["prev_pos"]
        )
        obs = self.get_obs(data, new_gate)
        done = self.is_terminated(data)
        reward = jax.lax.select(done, -20.0, reward)

        new_info = {
            "current_gate": new_gate,
            "prev_pos": curr_pos,
        }

        return state.replace(data=data, obs=obs, reward=reward, done=done, info=new_info)
    
    def get_obs(self, data: mjx.Data, current_gate: jax.Array) -> jax.Array:
        """Build observation vector.

        Args:
            data: MuJoCo simulation data.
            current_gate: Index of the current target gate (1,).

        Returns:
            Observation (19,): gyro, accel, quat, vel, target_dir, next_dir.
        """
        drone_pos = data.xpos[self.drone_body_id]
        gate_idx = current_gate[0].astype(jnp.int32)
        next_idx = (gate_idx + 1) % self.gate_count

        normalise = lambda q: q / (jnp.linalg.norm(q) + 1e-6)
        target_dir = normalise(self.gates["positions"][gate_idx] - drone_pos)
        next_dir = normalise(self.gates["positions"][next_idx] - drone_pos)

        # sensor data: gyro, accelerometer, and framequat
        return jnp.concatenate([data.sensordata[0:10], data.qvel[0:3], target_dir, next_dir])

    def check_gate_passage(
        self,
        data: mjx.Data,
        current_gate: jax.Array,
        prev_pos: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Check gate passage and compute reward.

        Args:
            data: MuJoCo simulation data.
            current_gate: Index of the current target gate (1,).
            prev_pos: Drone position from the previous timestep (3,).

        Returns:
            Tuple of (reward, new_gate_index, current_pos).
        """
        curr_pos = data.xpos[self.drone_body_id]
        gate_idx = current_gate[0].astype(jnp.int32)
        gate_pos = self.gates["positions"][gate_idx]
        gate_angle = self.gates["angles"][gate_idx]

        # transform to gate's local frame
        cos_a, sin_a = jnp.cos(gate_angle), jnp.sin(gate_angle)
        rel = curr_pos - gate_pos
        local_x = cos_a * rel[0] + sin_a * rel[1]
        local_y = -sin_a * rel[0] + cos_a * rel[1]
        local_z = rel[2] - 1.0

        in_gate = (jnp.abs(local_x) < 0.9) & (jnp.abs(local_y) < 0.3) & (jnp.abs(local_z) < 0.9)

        prev_dist = jnp.linalg.norm(prev_pos - gate_pos)
        curr_dist = jnp.linalg.norm(curr_pos - gate_pos)

        reward = jax.lax.select(in_gate, 1.0, 0.01 * jnp.sign(prev_dist - curr_dist))
        new_gate = jax.lax.select(in_gate, (current_gate + 1) % self.gate_count, current_gate)

        return reward, new_gate, curr_pos
    
    def is_terminated(self, data: mjx.Data) -> jax.Array:
        """Check whether the episode should terminate.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Boolean indicating termination.
        """
        drone_pos = data.xpos[self.drone_body_id]
        drone_quat = data.xquat[self.drone_body_id]

        has_collision = data.ncon > 0
        too_far = jnp.linalg.norm(drone_pos[:2]) > 50.0
        too_low = drone_pos[2] < 0.05

        up_z = 1.0 - 2.0 * (drone_quat[1] ** 2 + drone_quat[2] ** 2)
        flipped = up_z < 0.0

        return has_collision | too_far | too_low | flipped
