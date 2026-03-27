"""Stub for magi_compiler - replaces the Sand.ai JIT compiler with no-ops."""


def magi_compile(func=None, config_patch=None, **kwargs):
    """No-op decorator that just returns the class/function unchanged."""
    if func is not None:
        return func
    def wrapper(f):
        return f
    return wrapper
