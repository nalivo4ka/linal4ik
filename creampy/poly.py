from typing import Iterable, Union, get_args

import creampy
from creampy import Value, a_value_type, Tensor
from creampy.linalg import square_matrix_exception


class Polynom:
    def __init__(self, coefficients: Iterable[a_value_type]) -> None:
        self.coefficients = [
            v if isinstance(v, Value) else Value(v) for v in coefficients
        ]

    @property
    def derivative(self) -> 'Polynom':
        return Polynom([(i + 1) * c for i, c in enumerate(self.coefficients[1:])])

    @property
    def degree(self) -> int:
        return len(self.coefficients)

    def __call__(self, value: Union[Tensor, a_value_type]) -> Union[Tensor, a_value_type]:
        if isinstance(value, Tensor):
            square_matrix_exception(value)
            operation = lambda left, right: creampy.linalg.dot(left, right)
            initial = creampy.eye(value.shape)
            result = creampy.zeros(value.shape)
        elif isinstance(value, get_args(a_value_type)):
            operation = lambda left, right: left * right
            initial = 1
            result = 0
        else:
            raise ValueError("Expected Matrix or Number")

        for _power in range(self.degree):
            result += initial * self.coefficients[_power]
            initial = operation(initial, value)

        return result

    def __str__(self) -> str:
        return (f'{round(self.coefficients[0], 2)} + '
                + ' + '.join([f'({round(c, 2)})x^{i + 1}' for i, c in enumerate(self.coefficients[1:]) if c != 0])
                )

    # TODO: __repr__