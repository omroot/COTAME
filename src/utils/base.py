"""Base utility classes for the COTAME project."""

from typing import Any


class DotDict(dict):
    """Dictionary subclass that allows attribute-style access to items.

    Example:
        config = DotDict({'key': 'value'})
        config.key  # returns 'value'
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setattr__
    __delattr__ = dict.__delitem__


class Singleton(type):
    """Metaclass that ensures only one instance of a class exists.

    Usage:
        class MyClass(metaclass=Singleton):
            pass
    """

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
