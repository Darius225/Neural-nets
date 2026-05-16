"""Memoisation primitives.

Lives at the top of src/ (not under src/search/) because nothing here
is search-specific: the decorator just caches function results keyed
by an arbitrary projection of the first argument. The ES happens to be
the heaviest user today, but the same primitive is useful any time
you want :func:`functools.cache` semantics on an unhashable argument.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Hashable, TypeVar

T = TypeVar("T")


def memoize_by(key_fn: Callable[[Any], Hashable], *, enabled: bool = True):
    """Cache function results, keying by ``key_fn(first_arg)``.

    The decorated function exposes ``.hits``, ``.misses``, and
    ``.cache`` attributes for inspection. With ``enabled=False`` no
    caching happens and ``.misses`` simply counts every call — handy
    for benchmarking "with vs without cache" without changing call
    sites.

    Used because the natural argument (a dict / Pydantic model) is
    not hashable, so :func:`functools.cache` can't help directly.
    Pass a ``key_fn`` that projects the argument to something hashable
    (typically a sorted tuple of items).
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(arg, *rest, **kwargs):
            if not enabled:
                wrapper.misses += 1
                return fn(arg, *rest, **kwargs)
            key = key_fn(arg)
            if key in wrapper.cache:
                wrapper.hits += 1
                return wrapper.cache[key]
            wrapper.misses += 1
            result = fn(arg, *rest, **kwargs)
            wrapper.cache[key] = result
            return result

        wrapper.cache = {}
        wrapper.hits = 0
        wrapper.misses = 0
        return wrapper
    return decorator
