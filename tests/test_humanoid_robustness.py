"""Realized-dynamics validation with a small simulator-free Gym API double."""

import ast
import copy
import importlib.util
import json
import random
import subprocess
import sys
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "integrations/humanoid-gym/robustness.py"
MODULE_SPEC = importlib.util.spec_from_file_location("humanoid_robustness_test", SOURCE)
ROBUSTNESS = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(ROBUSTNESS)


def config():
    return types.SimpleNamespace(
        terrain=types.SimpleNamespace(mesh_type="plane", static_friction=0.6, dynamic_friction=0.6),
        noise=types.SimpleNamespace(add_noise=False),
        domain_rand=types.SimpleNamespace(
            randomize_friction=False,
            randomize_base_mass=False,
            randomize_pd_gains=False,
            push_robots=False,
            action_delay=False,
            action_noise=0.0,
        ),
    )


class Cache:
    """Supports the same [env_id, 0] indexing as NumPy and Torch caches."""

    def __init__(self, rows, initial):
        self.rows = [[initial] for _ in range(rows)]

    def __getitem__(self, key):
        return self.rows[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.rows[key[0]][key[1]] = value


def vector(x, y, z):
    return types.SimpleNamespace(x=x, y=y, z=z)


def matrix(rows):
    return types.SimpleNamespace(x=vector(*rows[0]), y=vector(*rows[1]), z=vector(*rows[2]))


def update_inertia(body, coefficient=1.0):
    mass = body.mass * coefficient
    body.invMass = 1.0 / body.mass
    body.inertia = matrix([[2 * mass, 0.25 * mass, 0], [0.25 * mass, 3 * mass, 0], [0, 0, 4 * mass]])
    determinant = 6 - 0.25**2
    body.invInertia = matrix(
        [
            [3 / determinant / mass, -0.25 / determinant / mass, 0],
            [-0.25 / determinant / mass, 2 / determinant / mass, 0],
            [0, 0, 0.25 / mass],
        ]
    )


def body(mass):
    result = types.SimpleNamespace(mass=mass, com=vector(0.1, -0.2, 0.3))
    # Deliberately unlike the post-setter inertia, as imported URDF values can be.
    update_inertia(result, coefficient=100)
    return result


class BaseTask:
    def _process_rigid_shape_props(self, props, env_id):
        self.super_shape_calls.append(env_id)
        if env_id == 0:
            self.friction_coeffs = Cache(self.num_envs, props[0].friction)
        return props

    def _process_rigid_body_props(self, props, env_id):
        self.super_body_calls.append(env_id)
        if env_id == 0:
            self._base_masses_cpu = Cache(self.num_envs, 0.0)
        self._base_masses_cpu[env_id, 0] = props[0].mass
        return props


class FakeGym:
    def __init__(self):
        self.bodies = {}
        self.shapes = {}
        self.names = {}

    def get_actor_rigid_body_names(self, handle, actor):
        return list(self.names[handle, actor])

    def get_actor_rigid_body_properties(self, handle, actor):
        return copy.deepcopy(self.bodies[handle, actor])

    def get_actor_rigid_shape_properties(self, handle, actor):
        return copy.deepcopy(self.shapes[handle, actor])


def environment(condition="nominal", num_envs=3):
    cfg = config()
    spec = ROBUSTNESS.configure_robustness(cfg, condition)
    registry = types.SimpleNamespace(task_classes={"task": BaseTask})
    cls = ROBUSTNESS.install_robustness_task(registry, "task", condition)
    env = cls()
    env.cfg = cfg
    env.num_envs = num_envs
    env.common_step_counter = 0
    env.super_shape_calls, env.super_body_calls = [], []
    env.envs = list(range(num_envs))
    env.actor_handles = [0] * num_envs
    env.gym = FakeGym()
    for env_id in range(num_envs):
        shapes = [types.SimpleNamespace(friction=value) for value in (0.7, 0.9)]
        props = [body(4), body(2)]
        shapes = env._process_rigid_shape_props(shapes, env_id)
        props = env._process_rigid_body_props(props, env_id)
        # Model the upstream set_actor_rigid_body_properties(recomputeInertia=True).
        for prop in props:
            update_inertia(prop)
        env.gym.shapes[env_id, 0] = copy.deepcopy(shapes)
        env.gym.bodies[env_id, 0] = copy.deepcopy(props)
        env.gym.names[env_id, 0] = ["pelvis", "leg"]
    env.base_masses = copy.deepcopy(env._base_masses_cpu)
    return env, spec


class RobustnessTest(unittest.TestCase):
    def test_stdlib_only_import_and_python38_syntax(self):
        ast.parse(SOURCE.read_text(), feature_version=(3, 8))
        script = (
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert not ({'torch','numpy','isaacgym'} & set(sys.modules))"
        )
        subprocess.run([sys.executable, "-I", "-S", "-c", script, str(SOURCE)], check=True)

    def test_nominal_does_not_modify_configuration_or_registered_class(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        spec = ROBUSTNESS.configure_robustness(cfg, "nominal")
        self.assertEqual(cfg, before)
        registry = types.SimpleNamespace(task_classes={"task": BaseTask})
        self.assertIs(ROBUSTNESS.install_robustness_task(registry, "task", "nominal"), BaseTask)
        self.assertIs(registry.task_classes["task"], BaseTask)
        self.assertEqual(spec["rigid_body_mass_scale"], 1.0)
        self.assertIsNone(spec["robot_shape_friction"])
        env, spec = environment()
        realized = ROBUSTNESS.capture_realized_dynamics(env, spec)
        self.assertEqual(realized["first_actor"]["shape_friction"], [0.7, 0.9])
        self.assertEqual(realized["first_actor"]["total_mass"], 6.0)
        self.assertEqual(realized["configured_ground"]["source"], "configured_terrain_parameters")
        json.dumps(realized, allow_nan=False)

    def test_rejects_uncontrolled_noise_and_dynamics_before_modification(self):
        for condition in ROBUSTNESS.CONDITIONS:
            for name in vars(config().domain_rand):
                cfg = config()
                setattr(cfg.domain_rand, name, 0.1 if name == "action_noise" else True)
                before = copy.deepcopy(cfg)
                with self.subTest(condition=condition, flag=name), self.assertRaises(ValueError):
                    ROBUSTNESS.configure_robustness(cfg, condition)
                self.assertEqual(cfg, before)
        for field, value in (("noise", True), ("terrain", "trimesh")):
            cfg = config()
            if field == "noise":
                cfg.noise.add_noise = value
            else:
                cfg.terrain.mesh_type = value
            with self.assertRaises(ValueError):
                ROBUSTNESS.configure_robustness(cfg, "nominal")
        with self.assertRaises(ValueError):
            ROBUSTNESS.configure_robustness(config(), "unknown")

    def test_low_friction_sets_both_ground_and_all_shapes_without_random_draws(self):
        rng_state = random.getstate()
        env, spec = environment("low_friction")
        realized = ROBUSTNESS.capture_realized_dynamics(env, spec)
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(env.super_shape_calls, [0, 1, 2])
        self.assertEqual(env.super_body_calls, [0, 1, 2])
        self.assertEqual(realized["first_actor"]["shape_friction"], [0.2, 0.2])
        self.assertEqual(realized["first_actor"]["total_mass"], 6.0)
        self.assertEqual(realized["configured_ground"]["static_friction"], 0.2)
        self.assertEqual(realized["configured_ground"]["dynamic_friction"], 0.2)
        self.assertTrue(realized["validation"]["observation_caches_consistent"])
        self.assertEqual(realized["validation"]["all_actors_checked"], 3)

    def test_payload_scales_every_link_and_compares_recomputed_inertia(self):
        nominal_env, nominal_spec = environment()
        payload_env, payload_spec = environment("payload_mass")
        nominal = ROBUSTNESS.capture_realized_dynamics(nominal_env, nominal_spec)
        payload = ROBUSTNESS.capture_realized_dynamics(payload_env, payload_spec)
        self.assertEqual(payload_env.super_body_calls, [0, 1, 2])
        self.assertAlmostEqual(payload["first_actor"]["total_mass"], 6.9)
        self.assertEqual(payload["first_actor"]["nominal_total_mass"], 6.0)
        self.assertEqual([item["nominal_mass"] for item in payload["first_actor"]["bodies"]], [4, 2])
        self.assertEqual(payload["validation"]["inertia_scaling"], "requires_nominal_comparison")
        checked = ROBUSTNESS.validate_mass_scaling(nominal, payload)
        self.assertTrue(checked["mass_and_inertia_scaling_verified"])
        self.assertTrue(checked["inverse_properties_scaling_verified"])
        self.assertTrue(checked["center_of_mass_unchanged"])
        self.assertEqual(checked["body_count"], 2)
        json.dumps({"spec": payload_spec, "realized": payload, "comparison": checked}, allow_nan=False)

    def test_install_is_idempotent_and_rejects_condition_stacking(self):
        registry = types.SimpleNamespace(task_classes={"task": BaseTask, "other": BaseTask})
        cls = ROBUSTNESS.install_robustness_task(registry, "task", "payload_mass")
        self.assertIs(ROBUSTNESS.install_robustness_task(registry, "task", "payload_mass"), cls)
        self.assertIs(registry.task_classes["other"], BaseTask)
        for condition in ("nominal", "low_friction"):
            with self.assertRaises(ValueError):
                ROBUSTNESS.install_robustness_task(registry, "task", condition)
        env, _ = environment("payload_mass")
        with self.assertRaisesRegex(ValueError, "twice"):
            env._process_rigid_body_props([body(4), body(2)], 0)

    def test_readback_checks_every_environment_shape_body_and_cache(self):
        cases = ("shape", "body_mass", "body_names", "inertia", "com", "cpu_cache", "cache", "friction_cache")
        for case in cases:
            with self.subTest(case=case):
                env, spec = environment("low_friction")
                if case == "shape":
                    env.gym.shapes[2, 0][1].friction = 0.3
                elif case == "body_mass":
                    env.gym.bodies[2, 0][1].mass *= 1.2
                    update_inertia(env.gym.bodies[2, 0][1])
                elif case == "body_names":
                    env.gym.names[2, 0][1] = "different_leg"
                elif case == "inertia":
                    env.gym.bodies[2, 0][1].inertia.x.y *= 1.1
                elif case == "com":
                    env.gym.bodies[2, 0][1].com.z += 0.1
                elif case == "cpu_cache":
                    env._base_masses_cpu[2, 0] += 1.0
                elif case == "cache":
                    env.base_masses[2, 0] += 1.0
                else:
                    env.friction_coeffs[2, 0] = 0.6
                with self.assertRaises(ValueError):
                    ROBUSTNESS.capture_realized_dynamics(env, spec)

    def test_capture_rejects_late_or_unapplied_perturbation(self):
        env, spec = environment("payload_mass")
        env.common_step_counter = 1
        with self.assertRaisesRegex(ValueError, "first environment step"):
            ROBUSTNESS.capture_realized_dynamics(env, spec)
        env.common_step_counter = 0
        del env._prism_nominal_body_masses[2]
        with self.assertRaisesRegex(ValueError, "pre-scaling"):
            ROBUSTNESS.capture_realized_dynamics(env, spec)
        env, _ = environment()
        with self.assertRaisesRegex(ValueError, "class"):
            ROBUSTNESS.capture_realized_dynamics(env, spec)

    def test_capture_rejects_bad_physical_values(self):
        for attribute, value in (("mass", 0), ("invMass", float("nan")), ("invMass", -1)):
            env, spec = environment()
            setattr(env.gym.bodies[2, 0][0], attribute, value)
            with self.subTest(attribute=attribute, value=value), self.assertRaises(ValueError):
                ROBUSTNESS.capture_realized_dynamics(env, spec)

    def test_payload_comparison_rejects_missed_inertia_update_or_other_changes(self):
        nominal_env, nominal_spec = environment()
        payload_env, payload_spec = environment("payload_mass")
        nominal = ROBUSTNESS.capture_realized_dynamics(nominal_env, nominal_spec)
        original = ROBUSTNESS.capture_realized_dynamics(payload_env, payload_spec)
        for field in (
            "mass",
            "nominal_mass",
            "inv_mass",
            "com",
            "inertia",
            "inv_inertia",
            "friction",
            "ground",
        ):
            altered = copy.deepcopy(original)
            item = altered["first_actor"]["bodies"][1]
            if field in ("mass", "nominal_mass", "inv_mass"):
                item[field] *= 1.1
            elif field == "com":
                item[field][0] += 0.1
            elif field in ("inertia", "inv_inertia"):
                item[field] = copy.deepcopy(nominal["first_actor"]["bodies"][1][field])
            elif field == "friction":
                altered["first_actor"]["shape_friction"][1] = 0.2
            else:
                altered["configured_ground"]["dynamic_friction"] = 0.2
            with self.subTest(field=field), self.assertRaises(ValueError):
                ROBUSTNESS.validate_mass_scaling(nominal, altered)
        altered = copy.deepcopy(original)
        altered["validation"]["all_actors_checked"] = 1
        with self.assertRaises(ValueError):
            ROBUSTNESS.validate_mass_scaling(nominal, altered)

    def test_small_inverse_inertia_entries_allow_float32_matrix_roundoff(self):
        nominal_env, nominal_spec = environment()
        payload_env, payload_spec = environment("payload_mass")
        nominal = ROBUSTNESS.capture_realized_dynamics(nominal_env, nominal_spec)
        payload = ROBUSTNESS.capture_realized_dynamics(payload_env, payload_spec)
        inverse = payload["first_actor"]["bodies"][0]["inv_inertia"]
        matrix_norm = max(abs(value) for row in inverse for value in row)
        # A near-zero entry has unbounded relative error but tiny matrix error.
        inverse[0][2] += matrix_norm * 0.5e-6
        result = ROBUSTNESS.validate_mass_scaling(nominal, payload)
        self.assertEqual(result["inverse_inertia_matrix_relative_tolerance"], 1e-6)
        # Matrix-relative validation still rejects a material off-diagonal error.
        inverse[0][2] += 1e-3
        with self.assertRaisesRegex(ValueError, "matrix error"):
            ROBUSTNESS.validate_mass_scaling(nominal, payload)


if __name__ == "__main__":
    unittest.main()
