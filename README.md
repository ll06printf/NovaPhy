# NovaPhy

A 3D physics engine for embodied intelligence applications (robotics, RL, sim-to-real).

## Features

- **Free Rigid Body Dynamics** — collision detection + Sequential Impulse contact solver (PGS) with warm starting, Coulomb friction, and Baumgarte stabilization
- **Articulated Body Dynamics** — Featherstone CRBA in reduced/generalized coordinates (FK, RNEA inverse dynamics, CRBA mass matrix, Cholesky forward dynamics)
- **Collision Detection** — Sweep-and-Prune broadphase + Box/Sphere/Plane narrowphase (including SAT box-box)
- **Joint Types** — Revolute (hinge), Free (6-DOF), Fixed
- **Python API** via pybind11 with Polyscope visualization
- **pip-installable** C++17 core via scikit-build-core

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- [vcpkg](https://vcpkg.io/) installed (default: `F:/vcpkg`)
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate novaphy

# Install NovaPhy
pip install -e .
```

### Verify

```python
import novaphy
print(novaphy.version())  # 0.1.0
```

### Run a Demo

```bash
python demos/demo_stack.py          # Box stacking
python demos/demo_pyramid.py        # Pyramid of boxes
python demos/demo_wall_break.py     # Sphere smashing a wall
python demos/demo_double_pendulum.py # Chaotic double pendulum
```

## Demos

| Demo | Description | Physics |
|------|-------------|---------|
| `demo_stack.py` | 3 boxes falling and stacking | Free body + contact solver |
| `demo_pyramid.py` | 4-3-2-1 box pyramid | Multi-body stacking |
| `demo_friction_ramp.py` | Boxes on 30° ramp, different friction | Coulomb friction |
| `demo_wall_break.py` | 5x5 wall hit by sphere | Dynamic collision |
| `demo_newtons_cradle.py` | 5 elastic spheres | Restitution + momentum |
| `demo_dominoes.py` | 20 dominoes chain reaction | Angular impulse |
| `demo_double_pendulum.py` | Chaotic 2-link pendulum | Featherstone CRBA |
| `demo_hinge.py` | Door swinging on hinge | Revolute joint |
| `demo_rope_bridge.py` | 10-segment rope | Multi-joint chain |
| `demo_joint_chain.py` | 6-link hanging chain | 3D articulation |

## Architecture

```
User (Python) -> ModelBuilder -> Model (immutable) -> World -> step(dt)
                                                       |-> Free bodies:  SAP -> Narrowphase -> Sequential Impulse
                                                       |-> Articulated:  FK -> RNEA(bias) -> CRBA(H) -> Cholesky -> Integrate
Visualization: Polyscope (Python-side per-frame mesh transform updates)
```

## Python API Overview

```python
import numpy as np
import novaphy

# Free body simulation
builder = novaphy.ModelBuilder()
builder.add_ground_plane(y=0.0)
body = novaphy.RigidBody.from_box(1.0, np.array([0.5, 0.5, 0.5]))
idx = builder.add_body(body, novaphy.Transform.from_translation(np.array([0, 5, 0])))
builder.add_shape(novaphy.CollisionShape.make_box(np.array([0.5, 0.5, 0.5]), idx))
world = novaphy.World(builder.build())
for _ in range(1000):
    world.step(1/120)

# Articulated body simulation
art = novaphy.Articulation()
joint = novaphy.Joint()
joint.type = novaphy.JointType.Revolute
joint.axis = np.array([0, 0, 1])
art.joints = [joint]
art.bodies = [novaphy.RigidBody.from_box(1.0, np.array([0.1, 0.5, 0.1]))]
art.build_spatial_inertias()
q = np.array([0.5])           # joint angles
qd = np.zeros(1)              # joint velocities
H = novaphy.mass_matrix_crba(art, q)     # mass matrix
qdd = novaphy.forward_dynamics(art, q, qd, np.zeros(1), np.array([0, -9.81, 0]))
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core | C++17 (float32 only) |
| Math | Eigen3 |
| Bindings | pybind11 |
| Build | CMake + scikit-build-core |
| C++ Deps | vcpkg (eigen3, gtest) |
| Visualization | Polyscope |

## Testing

```bash
pytest tests/python/ -v     # 38 tests
```

## License

MIT
