import jax
import jax.numpy as jnp
import mujoco
from jinja2 import Template
from importlib.resources import files


def load_scene(
    key: jax.Array,
    gate_count: int,
    course_radius: float,
    vertical_deviation: float,
    horizontal_deviation: float,
) -> tuple[mujoco.MjModel, dict[str, jax.Array]]:
    """Generate a randomised drone racing course.

    Args:
        key: JAX random key for gate placement noise.
        gate_count: Number of gates in the course.
        course_radius: Radius of the circular course layout.
        vertical_deviation: Maximum vertical noise for gate positions.
        horizontal_deviation: Maximum horizontal noise for gate positions.

    Returns:
        Tuple of (MuJoCo model, gate data dict with 'positions' and 'angles').
    """
    key1, key2 = jax.random.split(key)
    angles = jnp.linspace(0, 2 * jnp.pi, gate_count, endpoint=False)
    noise_h = jax.random.uniform(key1, (gate_count, 2), minval=-horizontal_deviation, maxval=horizontal_deviation)
    noise_v = jax.random.uniform(key2, (gate_count,), minval=-vertical_deviation, maxval=vertical_deviation)

    positions = jnp.stack([
        course_radius * jnp.cos(angles) + noise_h[:, 0],
        course_radius * jnp.sin(angles) + noise_h[:, 1],
        jnp.full(gate_count, 3.0) + noise_v
    ], axis=1)

    gates_xml = [
        {
            "pos": f"{positions[i, 0]} {positions[i, 1]} {positions[i, 2]}",
            "angle": float(jnp.degrees(angles[i]))
        }
        for i in range(gate_count)
    ]
    assets = files("assets")
    xml = Template((assets / "scene.xml.j2").read_text()).render(gates=gates_xml)
    mj_model = mujoco.MjModel.from_xml_string(xml, assets={
        "x2.obj": (assets / "x2.obj").read_bytes(),
        "x2_texture.png": (assets / "x2_texture.png").read_bytes()
    })

    gates = {"positions": positions, "angles": angles}
    return mj_model, gates
