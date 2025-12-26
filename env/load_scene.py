import jax
import jax.numpy as jnp
import mujoco
from jinja2 import Template
from importlib.resources import files


def load_scene(
    key: jax.Array,
    gate_count: int,
    drone_count: int,
    course_radius: float,
    vertical_deviation: float,
    horizontal_deviation: float
) -> mujoco.MjModel:
    k1, k2 = jax.random.split(key)
    angles = jnp.linspace(0, 2 * jnp.pi, gate_count, endpoint=False)
    noise_h = jax.random.uniform(k1, (gate_count, 2), minval=-horizontal_deviation, maxval=horizontal_deviation)
    noise_v = jax.random.uniform(k2, (gate_count,), minval=-vertical_deviation, maxval=vertical_deviation)
    gates = [
        {
            "pos": f"{course_radius * jnp.cos(a) + noise_h[i, 0]} {course_radius * jnp.sin(a) + noise_h[i, 1]} {3.0 + noise_v[i]}",
            "angle": float(jnp.degrees(a))
        }
        for i, a in enumerate(angles)
    ]
    drones = [{"pos": f"{i * 0.5} 0 0.1"} for i in range(drone_count)]

    assets = files("assets")
    xml = Template((assets / "scene.xml.j2").read_text()).render(gates=gates, drones=drones)
    return mujoco.MjModel.from_xml_string(xml, assets={
        "x2.obj": (assets / "x2.obj").read_bytes(),
        "x2_texture.png": (assets / "x2_texture.png").read_bytes()
    })
