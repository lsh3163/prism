"""Register released G1 tasks while preserving upstream task/config interfaces."""

from recipe_catalog import (
    DEFAULT_VARIANT_NAMES,
    PUBLIC_VARIANT_NAMES,
    describe_variant,
    recipe_for_variant,
    unique_recipes,
)


# Public compatibility mapping used by existing runners and manifests.  The
# catalog owns the values; this adapter intentionally retains the old shape.
VARIANTS = {
    name: (recipe_for_variant(name).task_name, recipe_for_variant(name).config_class)
    for name in PUBLIC_VARIANT_NAMES
}
DEFAULT_VARIANTS = DEFAULT_VARIANT_NAMES


def register_tasks():
    """Use the extracted actor and task, retaining upstream PPO update semantics.

    Isaac Gym must be imported by the caller before this function imports torch.
    RSL-RL v1.0.2 resolves policy and algorithm names in its runner module. Keep
    that upstream interface while making the imported implementation explicit.
    """
    import config
    from actor import ActorCritic, PolyActorCritic
    from g1_env import G1Robot
    from gated_actor import GatedPolyActorCritic
    from legged_gym.utils import task_registry
    from residual_gated_actor import ResidualLearnedGateActorCritic
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.ActorCritic = ActorCritic
    on_policy_runner.PolyActorCritic = PolyActorCritic
    on_policy_runner.GatedPolyActorCritic = GatedPolyActorCritic
    on_policy_runner.ResidualLearnedGateActorCritic = ResidualLearnedGateActorCritic

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

    for recipe in unique_recipes():
        task_registry.register(
            recipe.task_name,
            G1Robot,
            config.G1HumanoidGymCfg(),
            getattr(config, recipe.config_class)(),
        )
    return task_registry
