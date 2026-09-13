"""Register the released simulation tasks in this process, without editing upstream."""

VARIANTS = {
    "baseline": ("g1_humanoidgym_ppo", "G1HumanoidGymCfgPPO"),
    "larger": ("g1_humanoidgym_ppo_parammatch", "G1HumanoidGymCfgPPOParamMatched"),
    "degree1": ("g1_humanoidgym_ppo_poly_d1", "G1HumanoidGymCfgPPOPolyD1"),
    "degree2": ("g1_humanoidgym_ppo_poly", "G1HumanoidGymCfgPPOPoly"),
    "prism": ("g1_humanoidgym_ppo_poly_warmup", "G1HumanoidGymCfgPPOPolyWarmup"),
    "degree3": ("g1_humanoidgym_ppo_poly_d3", "G1HumanoidGymCfgPPOPolyD3"),
}


def register_tasks():
    """Use the extracted actor and task, retaining upstream PPO update semantics.

    Isaac Gym must be imported by the caller before this function imports torch.
    RSL-RL v1.0.2 resolves policy and algorithm names in its runner module. Keep
    that upstream interface while making the imported implementation explicit.
    """
    import config
    from actor import ActorCritic, PolyActorCritic
    from g1_env import G1Robot
    from legged_gym.utils import task_registry
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.ActorCritic = ActorCritic
    on_policy_runner.PolyActorCritic = PolyActorCritic

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

    for task_name, config_name in VARIANTS.values():
        task_registry.register(task_name, G1Robot, config.G1HumanoidGymCfg(), getattr(config, config_name)())
    return task_registry
