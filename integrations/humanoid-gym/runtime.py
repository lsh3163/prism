"""Register the released simulation tasks in this process, without editing upstream."""

VARIANTS = {
    "baseline": ("g1_humanoidgym_ppo", "G1HumanoidGymCfgPPO"),
    "larger": ("g1_humanoidgym_ppo_larger_v2", "G1HumanoidGymCfgPPOLarger"),
    "degree1": ("g1_humanoidgym_ppo_gated_d1", "G1HumanoidGymCfgPPOGatedD1"),
    "prism": ("g1_humanoidgym_ppo_gated_d2", "G1HumanoidGymCfgPPOGated"),
    "degree2": ("g1_humanoidgym_ppo_gated_d2", "G1HumanoidGymCfgPPOGated"),
    "degree3": ("g1_humanoidgym_ppo_gated_d3", "G1HumanoidGymCfgPPOGatedD3"),
    "legacy-larger": ("g1_humanoidgym_ppo_parammatch", "G1HumanoidGymCfgPPOParamMatched"),
    "legacy-degree1": ("g1_humanoidgym_ppo_poly_d1", "G1HumanoidGymCfgPPOPolyD1"),
    "legacy-degree2": ("g1_humanoidgym_ppo_poly", "G1HumanoidGymCfgPPOPoly"),
    "legacy-prism": ("g1_humanoidgym_ppo_poly_warmup", "G1HumanoidGymCfgPPOPolyWarmup"),
    "legacy-degree3": ("g1_humanoidgym_ppo_poly_d3", "G1HumanoidGymCfgPPOPolyD3"),
}
DEFAULT_VARIANTS = ("baseline", "larger", "degree1", "prism", "degree3")


def register_tasks():
    """Use the extracted actor and task, retaining upstream PPO update semantics.

    Isaac Gym must be imported by the caller before this function imports torch.
    RSL-RL v1.0.2 resolves policy and algorithm names in its runner module. Keep
    that upstream interface while making the imported implementation explicit.
    """
    import config
    from actor import ActorCritic, PolyActorCritic
    from gated_actor import GatedPolyActorCritic
    from g1_env import G1Robot
    from legged_gym.utils import task_registry
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.ActorCritic = ActorCritic
    on_policy_runner.PolyActorCritic = PolyActorCritic
    on_policy_runner.GatedPolyActorCritic = GatedPolyActorCritic

    # The original research checkout already has this hook. A clean upstream
    # checkout does not. Add it exactly once, after each completed PPO update.
    original_ppo = on_policy_runner.PPO
    if "advance_poly_warmup" not in original_ppo.update.__code__.co_names:

        class HumanoidPPO(original_ppo):
            def update(self):
                losses = super().update()
                if hasattr(self.actor_critic, "advance_poly_warmup"):
                    self.actor_critic.advance_poly_warmup()
                return losses

        on_policy_runner.PPO = HumanoidPPO

    for task_name, config_name in dict.fromkeys(VARIANTS.values()):
        task_registry.register(task_name, G1Robot, config.G1HumanoidGymCfg(), getattr(config, config_name)())
    return task_registry
