"""Noise schedulers for diffusion and sequence editing objectives."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import jax.numpy as jnp


def _as_array(value):
    return jnp.asarray(value, dtype=jnp.float32)


@dataclasses.dataclass
class BaseAlphaScheduler:
    __registry__: ClassVar[dict[str, type["BaseAlphaScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlphaScheduler.__registry__[cls.__name__] = cls
        BaseAlphaScheduler.__registry__[cls.__name__.lower()] = cls

    def __call__(self, t):
        return self.alpha(t)

    def alpha(self, t):
        t = _as_array(t)
        return self._alpha(t)

    def alpha_derivative(self, t):
        t = _as_array(t)
        return self._alpha_derivative(t)

    def reverse_mask_prob(self, s, t):
        s = _as_array(s)
        t = _as_array(t)
        return (1.0 - self.alpha(s)) / (1.0 - self.alpha(t) + 1e-6)

    def weight(self, t):
        return -self.alpha_derivative(t) / (1.0 - self.alpha(t) + 1e-6)

    def _alpha(self, t):
        raise NotImplementedError

    def _alpha_derivative(self, t):
        raise NotImplementedError


@dataclasses.dataclass
class LinearAlphaScheduler(BaseAlphaScheduler):
    def _alpha(self, t):
        return 1.0 - t

    def _alpha_derivative(self, t):
        return -jnp.ones_like(t)


@dataclasses.dataclass
class CosineAlphaScheduler(BaseAlphaScheduler):
    def _alpha(self, t):
        return 1.0 - jnp.cos((math.pi / 2.0) * (1.0 - t))

    def _alpha_derivative(self, t):
        return -(math.pi / 2.0) * jnp.sin((math.pi / 2.0) * (1.0 - t))


@dataclasses.dataclass
class BaseKappaScheduler:
    __registry__: ClassVar[dict[str, type["BaseKappaScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseKappaScheduler.__registry__[cls.__name__] = cls
        BaseKappaScheduler.__registry__[cls.__name__.lower()] = cls

    def __call__(self, t):
        return self.kappa(t)

    def kappa(self, t):
        t = _as_array(t)
        return self._kappa(t)

    def kappa_derivative(self, t):
        t = _as_array(t)
        return self._kappa_derivative(t)

    def weight(self, t):
        return self.kappa_derivative(t) / (1.0 - self.kappa(t) + 1e-6)

    def _kappa(self, t):
        raise NotImplementedError

    def _kappa_derivative(self, t):
        raise NotImplementedError


@dataclasses.dataclass
class CubicKappaScheduler(BaseKappaScheduler):
    a: float = 1.0
    b: float = 1.0

    def _kappa(self, t):
        return (self.a + 1.0) * (t**3) - (self.a + self.b + 1.0) * (t**2) + (self.b + 1.0) * t

    def _kappa_derivative(self, t):
        return 3.0 * (self.a + 1.0) * (t**2) - 2.0 * (self.a + self.b + 1.0) * t + (self.b + 1.0)


@dataclasses.dataclass
class LinearKappaScheduler(CubicKappaScheduler):
    a: float = -1.0
    b: float = 0.0


@dataclasses.dataclass
class CosineKappaScheduler(BaseKappaScheduler):
    def _kappa(self, t):
        return 1.0 - jnp.cos(0.5 * math.pi * t)

    def _kappa_derivative(self, t):
        return 0.5 * math.pi * jnp.sin(0.5 * math.pi * t)


def get_alpha_scheduler_class(name: str) -> type[BaseAlphaScheduler]:
    try:
        return BaseAlphaScheduler.__registry__[name]
    except KeyError:
        return BaseAlphaScheduler.__registry__[name.lower()]


def make_alpha_scheduler(name: str, **kwargs: Any) -> BaseAlphaScheduler:
    return get_alpha_scheduler_class(name)(**kwargs)


def get_kappa_scheduler_class(name: str) -> type[BaseKappaScheduler]:
    try:
        return BaseKappaScheduler.__registry__[name]
    except KeyError:
        return BaseKappaScheduler.__registry__[name.lower()]


def make_kappa_scheduler(name: str, **kwargs: Any) -> BaseKappaScheduler:
    return get_kappa_scheduler_class(name)(**kwargs)
