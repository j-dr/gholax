def __getattr__(name):
    if name == "Model":
        from .model import Model
        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
