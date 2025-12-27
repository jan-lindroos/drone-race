import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import brax
from brax.io import mjcf
from brax.envs.base import Env, PipelineEnv, State

from env.load_scene import load_scene


class DroneRace(PipelineEnv):
    """Drone racing environment with randomised gate courses."""

    def __init__(self, **kwargs):
        mj_model, self._gates = load_scene(jax.random.key(0))
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 10
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

    def reset(self, rng: jnp.ndarray) -> State:
        qpos = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        qvel = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pipeline_state = self.pipeline_init(qpos, qvel)

        reward, done = jnp.zeros(2)
        metrics = {}
        info = {
            "current_gate": jnp.zeros(1),
            "prev_pos": pipeline_state.x.pos[0],
        }
        obs = self._get_obs(pipeline_state, info["current_gate"])

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        # Scale action from [-1, 1] to actuator range [0, 13].
        ctrl = (action + 1.0) * 0.5 * 13.0
        new_pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        reward, new_gate, curr_pos = self._detect_gate_passage(
            new_pipeline_state, state.info["current_gate"], state.info["prev_pos"]
        )
        obs = self._get_obs(new_pipeline_state, new_gate)
        done = self._should_terminate(new_pipeline_state).astype(jnp.float32)
        reward = jax.lax.select(done > 0.5, -20.0, reward)

        new_info = state.info.copy()
        new_info["current_gate"] = new_gate
        new_info["prev_pos"] = curr_pos

        return state.replace(
            pipeline_state=new_pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
        )

    def _should_terminate(self, pipeline_state) -> jnp.ndarray:
        pos = pipeline_state.q[:3]
        is_out_of_bounds = jnp.linalg.norm(pos) > 20.0
        has_collision = jnp.any(pipeline_state.contact.dist < 0)
        
        return jnp.logical_or(is_out_of_bounds, has_collision)
    
    def _get_obs(self, pipeline_state, current_gate: jnp.ndarray) -> jnp.ndarray:
        """Build observation vector relative to drone body frame.

        Returns:
            Observation (21,): gate direction (3), gate normal (3),
            next gate direction (3), next gate normal (3),
            linear velocity (3), angular velocity (3), gravity vector (3).
        """
        drone_pos = pipeline_state.x.pos[0]
        drone_quat = pipeline_state.x.rot[0]
        get_body_frame = lambda vec: brax.math.inv_rotate(vec, drone_quat)

        gate_count = self._gates["positions"].shape[0]
        gate_idx = current_gate[0].astype(jnp.int32)
        next_idx = (gate_idx + 1) % gate_count

        def get_gate_obs(idx):
            gate_pos = self._gates["positions"][idx]
            gate_angle = self._gates["angles"][idx]

            direction = gate_pos - drone_pos
            norm_direction = direction / (jnp.linalg.norm(direction) + 1e-6)
            normal = jnp.array([jnp.cos(gate_angle), jnp.sin(gate_angle), 0.0])

            return get_body_frame(norm_direction), get_body_frame(normal)

        linear_velocity = get_body_frame(pipeline_state.xd.vel[0])
        angular_velocity = get_body_frame(pipeline_state.xd.ang[0])
        gravity = get_body_frame(jnp.array([0.0, 0.0, -1.0]))

        return jnp.concatenate([
            *get_gate_obs(gate_idx), 
            *get_gate_obs(next_idx),
            linear_velocity, 
            angular_velocity,
            gravity
        ])

    def _detect_gate_passage(
        self,
        pipeline_state,
        current_gate: jax.Array,
        prev_pos: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Check gate passage and compute reward.

        Args:
            pipeline_state: Brax pipeline state.
            current_gate: Index of the current target gate (1,).
            prev_pos: Drone position from the previous timestep (3,).

        Returns:
            Tuple of (reward, new_gate_index, current_pos).
        """
        curr_pos = pipeline_state.x.pos[0]
        gate_count = self._gates["positions"].shape[0]
        gate_idx = current_gate[0].astype(jnp.int32)
        gate_pos = self._gates["positions"][gate_idx]
        gate_angle = self._gates["angles"][gate_idx]

        # Transform to gate's local frame (rotation around z-axis).
        gate_quat = brax.math.quat_rot_axis(jnp.array([0.0, 0.0, 1.0]), gate_angle)
        rel = curr_pos - gate_pos
        local = brax.math.inv_rotate(rel, gate_quat)
        local_z = local[2] - 1.0

        in_gate = (jnp.abs(local[0]) < 0.9) & (jnp.abs(local[1]) < 0.3) & (jnp.abs(local_z) < 0.9)

        prev_dist = jnp.linalg.norm(prev_pos - gate_pos)
        curr_dist = jnp.linalg.norm(curr_pos - gate_pos)

        reward = jax.lax.select(in_gate, 1.0, 0.01 * jnp.sign(prev_dist - curr_dist))
        new_gate = jax.lax.select(in_gate, (current_gate + 1) % gate_count, current_gate)

        return reward, new_gate, curr_pos


brax.envs.register_environment("drone_race", DroneRace)
