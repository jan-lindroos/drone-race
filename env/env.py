from mujoco_playground._src.mjx_env import MjxEnv, State
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp


class Env(MjxEnv):
    def __init__(self):
        # Load your MJCF model
        mj_model = mujoco.MjModel.from_xml_path("your_robot.xml")
        
        # MJX-specific solver config (important for performance)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        
        super().__init__(mj_model)
        
    @property
    def action_size(self) -> int:
        """Dimension of the action space."""
        ...
    
    @property 
    def observation_size(self) -> int:
        """Dimension of the observation space."""
        ...
        
    def reset(self, rng: jax.Array) -> State:
        """Reset to initial state, possibly with randomisation."""
        rng, rng_qpos, rng_qvel = jax.random.split(rng, 3)
        
        # Set initial qpos/qvel (with optional noise)
        qpos = ...  # shape: (model.nq,)
        qvel = ...  # shape: (model.nv,)
        
        # Initialise the MJX pipeline
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        
        obs = self._get_obs(data)
        reward, done = jnp.zeros(2)
        metrics = {}
        
        return State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info={"rng": rng})
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment forward."""
        # Apply action and step physics
        data = state.data.replace(ctrl=action)
        data = mjx.step(self.mjx_model, data)
        
        # Compute reward, termination, observation
        obs = self._get_obs(data)
        reward = self._compute_reward(data, action)
        done = self._is_terminated(data)
        
        return state.replace(data=data, obs=obs, reward=reward, done=done)
    
    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Extract observation from simulation state."""
        # e.g., concatenate qpos, qvel, sensor readings
        ...
    
    def _compute_reward(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        """Compute scalar reward."""
        ...
    
    def _is_terminated(self, data: mjx.Data) -> jax.Array:
        """Check termination conditions (bool as float)."""
        ...
