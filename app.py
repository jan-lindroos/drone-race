import jax
import mujoco.viewer
from env.load_scene import load_scene


model, _ = load_scene(key=jax.random.key(0))
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
