"""Deterministic, evaluation-only G1 material and mass perturbations.

This module needs only Python 3.8's standard library. Install its task hooks
before ``make_env``: the pinned upstream creation loop calls both hooks before
simulation and passes body properties to the setter with recomputeInertia=True.
Ground friction is recorded from configuration; robot properties are read back
from every actor. Uniform link-mass scaling represents increased density at
fixed geometry, not an attached payload with a new center of mass.
"""

import math
from typing import Any, Dict

CONDITIONS = ("nominal", "low_friction", "payload_mass")
FRICTION = 0.2
MASS_SCALE = 1.15
REL_TOL = 1e-4
ABS_TOL = 1e-6
INVERSE_MATRIX_REL_TOL = 1e-6


def _condition(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown robustness condition: " + str(condition))
    return condition


def _assert_evaluation_config(cfg):
    flags = (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_pd_gains",
        "push_robots",
        "action_delay",
    )
    for name in flags:
        if getattr(cfg.domain_rand, name, False):
            raise ValueError("Robustness evaluation requires domain_rand." + name + " disabled")
    if float(getattr(cfg.domain_rand, "action_noise", 0.0)) != 0.0:
        raise ValueError("Robustness evaluation requires action_noise=0")
    if cfg.noise.add_noise:
        raise ValueError("Robustness evaluation requires observation noise disabled")
    if cfg.terrain.mesh_type != "plane":
        raise ValueError("Robustness evaluation requires the plane terrain")


def _finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite " + label)
    return number


def _same(actual, expected, label):
    actual = _finite(actual, label)
    expected = _finite(expected, label)
    if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        raise ValueError(f"{label}: actual {actual} != expected {expected}")


def configure_robustness(env_cfg: Any, condition: str) -> Dict[str, Any]:
    """Configure ground friction and return a JSON-safe dynamics specification.

    Call after the standard evaluation overrides. Every condition requires
    stochastic dynamics perturbations/noise to be disabled. Nominal and payload
    do not change configuration; hooks apply robot properties at asset creation.
    """
    _condition(condition)
    _assert_evaluation_config(env_cfg)
    if condition == "low_friction":
        env_cfg.terrain.static_friction = FRICTION
        env_cfg.terrain.dynamic_friction = FRICTION
    ground = {
        "static_friction": _finite(env_cfg.terrain.static_friction, "ground static friction"),
        "dynamic_friction": _finite(env_cfg.terrain.dynamic_friction, "ground dynamic friction"),
        "source": "configured_terrain_parameters",
    }
    if min(ground["static_friction"], ground["dynamic_friction"]) < 0.0:
        raise ValueError("Ground friction must be nonnegative")
    return {
        "schema_version": 1,
        "condition": condition,
        "ground": ground,
        "robot_shape_friction": FRICTION if condition == "low_friction" else None,
        "rigid_body_mass_scale": MASS_SCALE if condition == "payload_mass" else 1.0,
        "mass_scope": "all_robot_rigid_bodies",
        "inertia_update": "upstream_recomputeInertia_True",
        "payload_interpretation": "uniform_link_mass_scaling_at_fixed_geometry_and_COM",
    }


def install_robustness_task(task_registry: Any, task_name: str, condition: str) -> Any:
    """Install a process-local eval subclass; leave nominal registration intact.

    Reinstalling the same condition is idempotent. Reusing a perturbed task for a
    different condition is an error: callers must register fresh tasks instead
    of silently accumulating mass scaling or retaining material overrides.
    """
    _condition(condition)
    original = task_registry.task_classes[task_name]
    installed = getattr(original, "_prism_robustness_condition", None)
    if installed is not None:
        if installed != condition:
            raise ValueError("Task already configured for robustness condition " + installed)
        return original
    if condition == "nominal":
        return original

    class RobustnessTask(original):
        _prism_robustness_condition = condition

        def _process_rigid_shape_props(self, props, env_id):
            _assert_evaluation_config(self.cfg)
            props = super()._process_rigid_shape_props(props, env_id)
            if condition == "low_friction":
                for prop in props:
                    prop.friction = FRICTION
                # This cache becomes a privileged-observation feature later.
                self.friction_coeffs[env_id, 0] = FRICTION
            return props

        def _process_rigid_body_props(self, props, env_id):
            _assert_evaluation_config(self.cfg)
            props = super()._process_rigid_body_props(props, env_id)
            if not hasattr(self, "_prism_nominal_body_masses"):
                self._prism_nominal_body_masses = {}
            if env_id in self._prism_nominal_body_masses:
                raise ValueError("Body properties processed twice for environment " + str(env_id))
            self._prism_nominal_body_masses[env_id] = [float(prop.mass) for prop in props]
            if condition == "payload_mass":
                for prop in props:
                    prop.mass *= MASS_SCALE
                # Preserve the existing critic's base-mass feature semantics.
                self._base_masses_cpu[env_id, 0] = props[0].mass
            return props

    RobustnessTask.__name__ = original.__name__ + "Robustness_" + condition
    task_registry.task_classes[task_name] = RobustnessTask
    return RobustnessTask


def _vector(value, label):
    return [_finite(getattr(value, axis), label + "." + axis) for axis in ("x", "y", "z")]


def _matrix(value, label):
    # Store Mat33's x/y/z vectors in their API order, including off-diagonals.
    return [_vector(getattr(value, axis), label + "." + axis) for axis in ("x", "y", "z")]


def _cache_value(cache, env_id):
    value = cache[env_id, 0]
    return float(value.item() if hasattr(value, "item") else value)


def _body_record(prop, name, nominal_mass):
    body = {
        "name": name,
        "mass": _finite(prop.mass, name + " mass"),
        "nominal_mass": _finite(nominal_mass, name + " nominal mass"),
        "inv_mass": _finite(prop.invMass, name + " inverse mass"),
        "com": _vector(prop.com, name + " COM"),
        "inertia": _matrix(prop.inertia, name + " inertia"),
        "inv_inertia": _matrix(prop.invInertia, name + " inverse inertia"),
    }
    if min(body["mass"], body["nominal_mass"], body["inv_mass"]) <= 0.0:
        raise ValueError("Nonpositive rigid-body mass for " + name)
    _same(body["mass"] * body["inv_mass"], 1.0, name + " mass/inverse mass")
    for row in range(3):
        if body["inertia"][row][row] <= 0.0 or body["inv_inertia"][row][row] <= 0.0:
            raise ValueError("Nonpositive inertia diagonal for " + name)
    return body


def _same_sequence(actual, expected, label, scale=1.0):
    if len(actual) != len(expected):
        raise ValueError("Mismatched length: " + label)
    for index in range(len(actual)):
        if isinstance(expected[index], list):
            _same_sequence(actual[index], expected[index], label, scale)
        else:
            _same(actual[index], expected[index] * scale, label + "[" + str(index) + "]")


def _same_actor(actual, reference):
    if actual["body_names"] != reference["body_names"]:
        raise ValueError("Rigid-body names differ between environments")
    _same_sequence(actual["shape_friction"], reference["shape_friction"], "cross-environment friction")
    for index, body in enumerate(actual["bodies"]):
        other = reference["bodies"][index]
        for key in ("mass", "nominal_mass", "inv_mass"):
            _same(body[key], other[key], "cross-environment " + key)
        for key in ("com", "inertia", "inv_inertia"):
            _same_sequence(body[key], other[key], "cross-environment " + key)


def _scaled_inverse_matrix(actual, reference, label):
    """Allow float32 inversion cancellation in small off-diagonal entries.

    Bound maximum entry error by one part per million of the expected matrix's
    largest entry (or ABS_TOL). Entrywise relative error is unstable near zero;
    the matrix bound still rejects an omitted 15% scaling by a wide margin.
    """
    if len(actual) != 3 or len(reference) != 3:
        raise ValueError("Expected 3x3 inverse inertia: " + label)
    expected, observed = [], []
    for row in range(3):
        if len(actual[row]) != 3 or len(reference[row]) != 3:
            raise ValueError("Expected 3x3 inverse inertia: " + label)
        for col in range(3):
            expected.append(_finite(reference[row][col], label) / MASS_SCALE)
            observed.append(_finite(actual[row][col], label))
    limit = max(ABS_TOL, INVERSE_MATRIX_REL_TOL * max(abs(value) for value in expected))
    error = max(abs(observed[index] - expected[index]) for index in range(9))
    if error > limit:
        raise ValueError(label + " inverse inertia matrix error " + str(error) + " exceeds " + str(limit))


def capture_realized_dynamics(env: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Read and validate every robot before the first environment step.

    Only the first actor's full properties are retained in JSON. All actors are
    checked against it and against the mass/friction caches used by the critic.
    Payload mass ratios use pre-setter hook measurements. Inertia scaling needs
    a separate nominal readback: URDF inertia can change during initial setup.
    """
    condition = _condition(spec["condition"])
    _assert_evaluation_config(env.cfg)
    if int(getattr(env, "common_step_counter", 0)) != 0:
        raise ValueError("Capture dynamics before the first environment step")
    expected_scale = MASS_SCALE if condition == "payload_mass" else 1.0
    _same(spec["rigid_body_mass_scale"], expected_scale, "specified mass scale")
    expected_friction = FRICTION if condition == "low_friction" else None
    if spec["robot_shape_friction"] != expected_friction:
        raise ValueError("Incorrect specified robot friction")
    if getattr(type(env), "_prism_robustness_condition", None) != (
        None if condition == "nominal" else condition
    ):
        raise ValueError("Environment class does not match robustness condition")
    for key in ("static_friction", "dynamic_friction"):
        value = getattr(env.cfg.terrain, key)
        _same(value, spec["ground"][key], "configured ground " + key)
        if condition == "low_friction":
            _same(value, FRICTION, "low-friction ground " + key)
    num_envs = int(env.num_envs)
    if num_envs <= 0 or len(env.envs) != num_envs or len(env.actor_handles) != num_envs:
        raise ValueError("Inconsistent environment/actor counts")
    nominal_masses = getattr(env, "_prism_nominal_body_masses", {})
    first = None
    for env_id in range(num_envs):
        handle, actor = env.envs[env_id], env.actor_handles[env_id]
        names = list(env.gym.get_actor_rigid_body_names(handle, actor))
        props = env.gym.get_actor_rigid_body_properties(handle, actor)
        shapes = env.gym.get_actor_rigid_shape_properties(handle, actor)
        if not props or len(props) != len(names) or not shapes:
            raise ValueError("Missing or inconsistent rigid-body/shape properties")
        if condition != "nominal" and env_id not in nominal_masses:
            raise ValueError("Missing pre-scaling body masses for environment " + str(env_id))
        reference_masses = nominal_masses.get(env_id, [float(prop.mass) for prop in props])
        if len(reference_masses) != len(props):
            raise ValueError("Missing pre-scaling body mass")
        bodies = [
            _body_record(prop, names[index], reference_masses[index]) for index, prop in enumerate(props)
        ]
        friction = [_finite(prop.friction, "robot shape friction") for prop in shapes]
        if min(friction) < 0.0:
            raise ValueError("Negative robot shape friction")
        for value in friction:
            if expected_friction is not None:
                _same(value, expected_friction, "low-friction robot shape")
        for body in bodies:
            _same(body["mass"], body["nominal_mass"] * expected_scale, body["name"] + " scaled mass")
        for key in ("_base_masses_cpu", "base_masses"):
            _same(_cache_value(getattr(env, key), env_id), bodies[0]["mass"], key + " cache")
        _same(_cache_value(env.friction_coeffs, env_id), friction[0], "friction_coeffs cache")
        current = {
            "body_names": names,
            "shape_friction": friction,
            "bodies": bodies,
            "total_mass": sum(body["mass"] for body in bodies),
            "nominal_total_mass": sum(body["nominal_mass"] for body in bodies),
        }
        if first is None:
            first = current
        else:
            _same_actor(current, first)
    return {
        "schema_version": 1,
        "condition": condition,
        "num_environments": num_envs,
        "first_actor": first,
        "all_envs_consistent": True,
        "configured_ground": dict(spec["ground"]),
        "validation": {
            "all_actors_checked": num_envs,
            "mass_scale": expected_scale,
            "mass_scale_verified": True,
            "mass_reference": "pre_scaling_hook" if condition != "nominal" else "nominal_readback",
            "robot_shape_friction_verified": condition == "low_friction",
            "observation_caches_consistent": True,
            "inertia_scaling": "requires_nominal_comparison" if condition == "payload_mass" else "not_scaled",
            "ground_properties": "configured_only_no_simulator_readback",
        },
    }


def validate_mass_scaling(nominal: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Verify realized mass/inertia x1.15 and unchanged COM/materials vs nominal.

    Inverse properties must scale reciprocally. Each input must already have
    verified every environment; this comparison uses their common first actor.
    """
    if nominal["condition"] != "nominal" or payload["condition"] != "payload_mass":
        raise ValueError("Expected nominal and payload_mass dynamics records")
    for record in (nominal, payload):
        if record["all_envs_consistent"] is not True:
            raise ValueError("All environments must have consistent realized dynamics")
        if record["validation"]["all_actors_checked"] != record["num_environments"]:
            raise ValueError("Dynamics capture did not verify every actor")
    first, second = nominal["first_actor"], payload["first_actor"]
    if first["body_names"] != second["body_names"] or len(first["bodies"]) != len(second["bodies"]):
        raise ValueError("Nominal/payload rigid bodies differ")
    _same_sequence(second["shape_friction"], first["shape_friction"], "payload shape friction")
    for key in ("static_friction", "dynamic_friction"):
        _same(payload["configured_ground"][key], nominal["configured_ground"][key], "payload ground " + key)
    for index, body in enumerate(second["bodies"]):
        reference = first["bodies"][index]
        for key, scale in (("mass", MASS_SCALE), ("nominal_mass", 1.0), ("inv_mass", 1.0 / MASS_SCALE)):
            _same(body[key], reference[key] * scale, body["name"] + " payload " + key)
        for key, scale in (("com", 1.0), ("inertia", MASS_SCALE)):
            _same_sequence(body[key], reference[key], body["name"] + " payload " + key, scale)
        _scaled_inverse_matrix(body["inv_inertia"], reference["inv_inertia"], body["name"])
    return {
        "mass_scale": MASS_SCALE,
        "mass_and_inertia_scaling_verified": True,
        "inverse_properties_scaling_verified": True,
        "center_of_mass_unchanged": True,
        "robot_and_ground_friction_unchanged": True,
        "body_count": len(first["bodies"]),
        "relative_tolerance": REL_TOL,
        "absolute_tolerance": ABS_TOL,
        "inverse_inertia_comparison": "maximum_entry_error_relative_to_expected_matrix_max_norm",
        "inverse_inertia_matrix_relative_tolerance": INVERSE_MATRIX_REL_TOL,
        "inverse_inertia_matrix_absolute_tolerance": ABS_TOL,
    }
