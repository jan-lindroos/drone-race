import jax
import mujoco.viewer
from env.load_scene import load_scene


model = load_scene(
    key=jax.random.key(0),
    gate_count=6,
    course_radius=12.0,
    vertical_deviation=2.0,
    horizontal_deviation=3.0,
)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
