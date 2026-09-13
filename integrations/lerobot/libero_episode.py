"""Episode termination semantics for LIBERO's success-valued done signal."""


def should_terminate_episode(backend_done: bool, is_success: bool, terminate_on_success: bool) -> bool:
    """Honor success continuation while preserving other backend termination."""
    if is_success:
        return bool(terminate_on_success)
    return bool(backend_done)
