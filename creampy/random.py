import random

import creampy
from creampy import Tensor


def seed(value: int) -> None:
    random.seed(value)


def _random(shape: tuple[int, ...]) -> Tensor:
    return creampy.full_array(shape, random.random)