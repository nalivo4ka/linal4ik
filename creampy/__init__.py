from functools import total_ordering, lru_cache
import itertools
from typing import Union, Callable, Optional, Iterable, get_args

value_type = Union[float, int, complex, bool, None]
EPS = 1e-8


@total_ordering
class Value:
    def __init__(self, value: value_type):
        if isinstance(value, complex):
            self.value = complex(round(value.real, 8), round(value.imag, 8))
        elif isinstance(value, Value):
            self.value = value.value
        elif self.is_value(value):
            self.value = value
        else:
            raise ValueError('Unexpected value')

    def __round__(self, n=None):
        if isinstance(self.value, get_args(Union[int, float])):
            return round(self.value, n)
        return self.value

    def __str__(self) -> str:
        if isinstance(self.value, bool):
            return 'true' if self.value else 'false'
        if isinstance(self.value, complex):
            return str(self.value)[1:-1] if '(' in str(self.value) else str(self.value)
        if self.value is None:
            return 'nan'
        return str(round(self.value, 8))

    @property
    def value_str_length(self) -> int:
        return len(str(self))

    def cast_str(self, required_length: Optional[int]) -> str:
        if required_length is None:
            required_length = self.value_str_length

        if required_length < self.value_str_length:
            raise ValueError("Wrong required length: < self length")

        result = " " * (required_length - self.value_str_length) + str(self)

        return result


    @staticmethod
    def is_value(value) -> bool:
        return isinstance(value, get_args(value_type))

    __repr__ = __str__

    def __bin_operation(
            self, other: Union['Value', value_type], operation: Callable[[value_type, value_type], value_type]
    ) -> 'Value':
        if isinstance(other, get_args(value_type)):
            return Value(operation(self.value, other))
        elif isinstance(other, Value):
            return Value(operation(self.value, other.value))
        else:
            raise ValueError('Unexpected argument in binary operation')

    def custom_eq(self, other: Union['Value', value_type], eps: float):
        if isinstance(other, get_args(value_type)):
            return abs(self.value - other) < eps
        elif isinstance(other, Value):
            return abs(self.value - other.value) < eps
        else:
            raise ValueError('Unexpected argument in operation')

    def __eq__(self, other: Union['Value', value_type]) -> bool:
        return self.custom_eq(other, EPS)

    def __lt__(self, other: Union['Value', value_type]) -> bool:
        if isinstance(other, get_args(value_type)):
            return self.value <= other
        elif isinstance(other, Value):
            return self.value <= other.value
        else:
            raise ValueError('Unexpected argument in operation')

    def __add__(self, other: Union['Value', value_type]) -> 'Value':
        try:
            return self.__bin_operation(other, lambda x, y: x + y)
        except ValueError:
            return other.__radd__(self)

    def __iadd__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__add__(other).value
        return self

    __radd__ = __add__

    def __sub__(self, other: Union['Value', value_type]) -> 'Value':
        try:
            return self.__bin_operation(other, lambda x, y: x - y)
        except ValueError:
            return other.__rsub__(self)

    def __isub__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__sub__(other).value
        return self

    def __rsub__(self, other: Union['Value', value_type]) -> 'Value':
        return self.__bin_operation(other, lambda x, y: y - x)

    def __mul__(self, other: Union['Value', value_type]) -> 'Value':
        try:
            return self.__bin_operation(other, lambda x, y: x * y)
        except ValueError:
            return other.__rmul__(self)

    def __imul__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__mul__(other).value
        return self

    __rmul__ = __mul__

    def __truediv__(self, other: Union['Value', value_type]) -> 'Value':
        if other == 0:
            if self.__gt__(0):
                return Value(float('inf'))
            return Value(-float('inf'))

        try:
            return self.__bin_operation(other, lambda x, y: x / y)
        except ValueError:
            return other.__rtruediv__(self)

    def __itruediv__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__truediv__(other).value
        return self

    def __rtruediv__(self, other: Union['Value', value_type]) -> 'Value':
        if self == 0:
            if other >= 0:
                return Value(float('inf'))
            return Value(-float('inf'))

        return self.__bin_operation(other, lambda x, y: y / x)

    def __pow__(self, power: Union['Value', value_type], modulo=None) -> 'Value':
        try:
            return self.__bin_operation(power, lambda x, y: x ** y)
        except ValueError:
            return power.__rpow__(self)

    def __ipow__(self, power: Union['Value', value_type]) -> 'Value':
        self.value = self.__ipow__(power).value
        return self

    def __rpow__(self, other: Union['Value', value_type]) -> 'Value':
        return self.__bin_operation(other, lambda x, y: y ** x)

    def __or__(self, other: Union['Value', value_type]) -> 'Value':
        return self.__bin_operation(other, lambda x, y: x | y)

    def __ior__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__or__(other)
        return self

    __ror__ = __or__

    def __and__(self, other: Union['Value', value_type]) -> 'Value':
        return self.__bin_operation(other, lambda x, y: x & y)

    def __iand__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__and__(other)
        return self

    __rand__ = __and__

    def __xor__(self, other: Union['Value', value_type]) -> 'Value':
        return self.__bin_operation(other, lambda x, y: x ^ y)

    def __ixor__(self, other: Union['Value', value_type]) -> 'Value':
        self.value = self.__xor__(other)
        return self

    __rxor__ = __xor__

    def __neg__(self) -> 'Value':
        return Value(-self.value)

    def __invert__(self):
        if isinstance(self.value, bool):
            return Value(self.value ^ True)

        return Value(~self.value)

    def copy(self) -> 'Value':
        return Value(self.value)

    def __hash__(self):
        return hash(self.value)

    def isna(self):
        return self.value is None

    def __bool__(self):
        return self.value


a_value_type = Union[Value, value_type]


class Tensor:
    def __init__(self, array, check_complex: bool=False):
        try:
            array[0]
        except Exception:
            raise ValueError("Unexpected value in Tensor")

        if Value.is_value(array[0]) or isinstance(array[0], Value):
            self.array: list[Union[Tensor, Value]] = [i if isinstance(i, Value) else Value(i) for i in array]
        else:
            self.array: list[Union[Tensor, Value]] = [
                cur_array if isinstance(cur_array, Tensor) else Tensor(cur_array) for cur_array in array
            ]

        if not self.__sanitize(self):
            raise ValueError("Wrong shape of Tensor")

        if not check_complex:
            return

        is_complex = False
        for index in Tensor.__static_full_search(self.shape):
            if isinstance(self[index].value, complex):
                is_complex = True
                break

        if not is_complex:
            return

        for index in Tensor.__static_full_search(self.shape):
            self[index] = Value(complex(self[index].value, 0))

    def to_complex(self) -> 'Tensor':
        _copy = self.copy()
        for index in Tensor.__static_full_search(self.shape):
            _copy[index] = Value(complex(_copy[index].value, 0))

        return _copy

    def __len__(self) -> int:
        return len(self.array)

    def copy(self) -> 'Tensor':
        if self.n_dims == 1:
            return Tensor([Value(i.value) for i in self.array])

        return Tensor([array.copy() for array in self.array])

    @property
    def dtype(self) -> type:
        return bool if isinstance(self[tuple([0] * self.n_dims)].value, bool) else Value

    @property
    def n_dims(self) -> int:
        return len(self.shape)

    @staticmethod
    def __get_shape(array: Union['Tensor', Value], prev_shape: list[int]) -> list[int]:
        if isinstance(array, Value):
            return prev_shape

        return Tensor.__get_shape(array.array[0], prev_shape + [len(array)])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.__get_shape(self, []))

    @staticmethod
    def __number_values(shape: tuple[int, ...]) -> int:
        answer = 1
        for cur_dim in shape:
            answer *= cur_dim

        return answer

    @property
    def number_values(self) -> int:
         return self.__number_values(self.shape)

    def to_list(self) -> list:
        if len(self.shape) == 1:
            return [value.value for value in self.array]

        return [array.to_list() for array in self.array]

    @staticmethod
    def __sanitize(
            array: 'Tensor', expected_shape: Optional[tuple[int, ...]] = None, cur_depth: Optional[int] = None
    ) -> bool:
        if expected_shape is None:
            expected_shape = array.shape
        if cur_depth is None:
            cur_depth = 0

        if cur_depth == len(expected_shape):
            return True

        if len(array) != expected_shape[cur_depth]:
            return False

        return all(Tensor.__sanitize(cur_array, expected_shape, cur_depth + 1) for cur_array in array.array)

    def __getitem__(
            self, item: Union[tuple[Union[slice, int, Iterable], ...], slice, int, Iterable, 'Tensor']
    ) -> Union['Tensor', Value]:
        if isinstance(item, Tensor) and item.dtype == bool:
            if item.shape != self.shape:
                raise IndexError("Wrong index!")
            result = []
            for index in Tensor.__static_full_search(self.shape):
                if item[index]:
                    result.append(self[index])
            return Tensor(result)

        if not isinstance(item, tuple):
            item = (item,)

        if len(item) == 0:
            return self

        cur_index = item[0]
        next_indexes = item[1:]
        if isinstance(cur_index, int):
            if isinstance(self.array[cur_index], Tensor):
                return self.array[cur_index].__getitem__(next_indexes)
            return self.array[cur_index]

        if isinstance(cur_index, slice):
            start = cur_index.start if cur_index.start is not None else 0
            stop = cur_index.stop if (cur_index.stop is not None and cur_index.stop != -1) else len(self)
            step = cur_index.step if cur_index.step is not None else 1
            cur_index = range(start, stop, step)

        if isinstance(cur_index, Tensor) and cur_index.n_dims == 1:
            cur_index = [value.value for value in cur_index.array]

        if isinstance(cur_index, Iterable):
            return Tensor([
                self.array[i].__getitem__(next_indexes) if isinstance(self.array[i], Tensor) else self.array[i]
                for i in cur_index
            ])

        raise IndexError("Wrong indexes")

    @staticmethod
    def __static_full_search(shape: tuple[int, ...]) -> Iterable[tuple[int]]:
        # TODO: Add lru_cache
        ranges = [range(dim_shape) for dim_shape in shape]
        return iter(itertools.product(*ranges))

    def count(self, value: a_value_type) -> int:
        answer = 0
        for index in self.__static_full_search(self.shape):
            if self[index] == value:
                answer += 1

        return answer

    def reshape(self, *new_shape: int) -> 'Tensor':
        if -1 not in new_shape:
            assert Tensor.__number_values(new_shape) == self.number_values
        else:
            assert new_shape.count(-1) == 1
            n = -Tensor.__number_values(new_shape)
            new_shape = list(new_shape)
            assert self.number_values % n == 0
            extra_dim = self.number_values // n
            new_shape[new_shape.index(-1)] = extra_dim
            new_shape = tuple(new_shape)

        result = zeros(new_shape)
        for new_index, old_index in zip(Tensor.__static_full_search(new_shape), Tensor.__static_full_search(self.shape)):
            result[new_index] = self[old_index].copy()

        return result

    @staticmethod
    def __permute_shape(shape: tuple[int, ...], new_dims: tuple[int, ...], check: bool = False) -> tuple[int, ...]:
        if check and list(sorted(list(new_dims))) != list(range(len(shape))):
            raise ValueError(f"Wrong dims: shape = {shape}, new_dims = {new_dims}")

        return tuple([shape[cur_dim] for cur_dim in new_dims])

    def permute(self, *new_dims: int) -> 'Tensor':
        new_shape = Tensor.__permute_shape(self.shape, new_dims, check=True)
        result = zeros(new_shape)
        for index in Tensor.__static_full_search(self.shape):
            result[Tensor.__permute_shape(index, new_dims)] = self[index]

        return result

    @property
    def T(self) -> 'Tensor':
        if len(self.shape) != 2:
            raise ValueError("Excepted matrix")

        return self.permute(1, 0)

    def __setitem__(
            self, key: Union[tuple[Union[slice, int, Iterable], ...], slice, int, Iterable, 'Tensor'],
            value: Union['Tensor', a_value_type, list],
    ):
        subarray = self[key]
        if isinstance(subarray, Value):
            if not isinstance(value, get_args(a_value_type)):
                raise ValueError("Bad value to set!")
            value = Value(value)
            subarray.value = value.value
            return

        if isinstance(value, list):
            value = Tensor(value)

        if isinstance(value, Tensor):
            if value.shape != subarray.shape:
                raise ValueError("Bad value to set!")

            for index in Tensor.__static_full_search(subarray.shape):
                subarray[index].value = value[index].value

        elif isinstance(value, get_args(a_value_type)):
            value = Value(value)
            for index in  Tensor.__static_full_search(subarray.shape):
                subarray[index].value = value.value

        else:
            raise ValueError("Bad value to set!")

    def update(self, array: 'Tensor') -> None:
        assert self.shape == array.shape
        for index in  Tensor.__static_full_search(self.shape):
            self[index] = array[index]

    def __iter__(self) -> Iterable[Union['Tensor', Value]]:
        # TODO: __iter__
        return self.array

    @staticmethod
    @lru_cache
    def __custom_str(array: 'Tensor', value_length: Optional[int] = None) -> str:
        if value_length is None:
            lengths_array = [array[index].value_str_length for index in Tensor.__static_full_search(array.shape)]
            value_length = max(lengths_array)

        if array.n_dims == 1:
            return '[' + ' '.join(map(lambda value: value.cast_str(value_length), array.array)) + ']'

        inside_strings = "\t" + "\n".join(map(
            lambda _array: Tensor.__custom_str(_array, value_length), array.__iter__()
        ))
        inside_strings = inside_strings.replace('\n', '\n\t')

        return f'[\n{inside_strings}\t\n]'

    def __str__(self) -> str:
        return Tensor.__custom_str(self)

    def main_diagonal(self) -> 'Tensor':
        trace_array = []
        for index in range(min(self.shape)):
            trace_array.append(self.__getitem__(tuple([index] * self.n_dims)))

        return Tensor(trace_array)

    @property
    def trace(self) -> a_value_type:
        return self.main_diagonal().sum()

    def flatten(self) -> 'Tensor':
        if self.n_dims == 1:
            return self

        return concatenate(*[array.flatten() for array in self.array])

    def __hash__(self):
        return hash(tuple(self.array))

    def clean(self, eps: float = EPS) -> None:
        for index in Tensor.__static_full_search(self.shape):
            cur_value = self[index]
            if isinstance(cur_value.value, bool):
                continue
            if isinstance(cur_value.value, complex):
                real = Value(cur_value.value.real)
                imag = Value(cur_value.value.imag)
                if real.custom_eq(round(real), eps):
                    real = round(real.value)
                else:
                    real = real.value
                if imag.custom_eq(round(imag), eps):
                    imag = round(imag.value)
                else:
                    imag = imag.value
                self[index] = complex(real, imag)

            elif cur_value.custom_eq(round(cur_value.value), eps):
                self[index] = Value(round(cur_value.value))

    def isna(self) -> 'Tensor':
        mask = full_array(self.shape, False)
        for index in Tensor.__static_full_search(self.shape):
            if self[index].isna():
                mask[index] = True

        return mask

    @staticmethod
    def __by_elementary_operation(
            left: 'Tensor', value: a_value_type, operation: Callable[[a_value_type, a_value_type], a_value_type]
    ) -> 'Tensor':
        result = zeros(left.shape)
        for index in Tensor.__static_full_search(left.shape):
            result[index] = operation(left[index], value)

        return result

    @staticmethod
    def __array_operation(
            left: 'Tensor', other: 'Tensor',
            operation: Callable[[a_value_type, a_value_type], a_value_type],
    ) -> 'Tensor':
        if left.shape != other.shape:
            raise ValueError("Arrays have different shapes")

        result = zeros(left.shape)
        for index in Tensor.__static_full_search(left.shape):
            result[index] = operation(left[index], other[index])

        return result

    @staticmethod
    def __any_operation(
            left: 'Tensor', other: Union['Tensor', a_value_type],
            operation: Callable[[a_value_type, a_value_type], a_value_type],
    ) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor.__array_operation(left, other, operation)
        elif isinstance(other, get_args(a_value_type)):
            return Tensor.__by_elementary_operation(left, other, operation)
        else:
            raise ValueError(f"Bad argument for operation: {other}")

    def __add__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x + y)

    def __iadd__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__add__(other).array
        return self

    def __radd__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y + x)

    def __sub__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x - y)

    def __isub__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__sub__(other).array
        return self

    def __rsub__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y - x)

    def __mul__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x * y)

    def __imul__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__mul__(other).array
        return self

    def __rmul__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: x * y)

    def __truediv__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x / y)

    def __itruediv__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__truediv__(other).array
        return self

    def __rtruediv__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y / x)

    def __pow__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x ** y)

    def __ipow__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__pow__(other).array
        return self

    def __rpow__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y ** x)

    def __or__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x | y)

    def __ior__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__or__(other).array
        return self

    def __ror__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y | x)

    def __and__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x & y)

    def __iand__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__and__(other).array
        return self

    def __rand__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y & x)

    def __xor__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        return Tensor.__any_operation(self, other, lambda x, y: x ^ y)

    def __ixor__(self, other: Union['Tensor', a_value_type]) -> 'Tensor':
        self.array = self.__xor__(other).array
        return self

    def __rxor__(self, other: a_value_type) -> 'Tensor':
        return Tensor.__by_elementary_operation(self, other, lambda x, y: y ^ x)

    def __neg__(self) -> 'Tensor':
        return self * (-1)

    def __invert__(self) -> 'Tensor':
        result = self.copy()
        for index in Tensor.__static_full_search(result.shape):
            result[index] = ~result[index]

        return result

    def __matmul__(self, other: 'Tensor') -> Union['Tensor', a_value_type]:
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrices must satisfy the dimension condition."
                f" Given {self.shape} and {other.shape}, {self.shape[1]}!={other.shape[0]}"
            )

        result = zeros((self.shape[0], other.shape[1]))
        for y in range(self.shape[0]):
            for x in range(other.shape[1]):
                result[y, x] = (self[y] * other[:, x]).sum()

        if result.shape[0] * result.shape[1] == 1:
            return result[0][0]

        return result


    @staticmethod
    def __aggregation_operation(
            array: 'Tensor',
            operation: Callable[[a_value_type, a_value_type], a_value_type],
            default: Union['Tensor', a_value_type],
            axis: Union[None, int, tuple[int, ...]] = None,
    ) -> Union['Tensor', value_type]:
        if isinstance(axis, int):
            axis = (axis,)

        if axis is None or len(axis) == array.n_dims:
            if isinstance(default, Tensor):
                raise ValueError("Wrong default arg in agg operation")
            for index in Tensor.__static_full_search(array.shape):
                default = operation(default, array[index])
            return default.value if isinstance(default, Value) else default

        target_shape = tuple([cur_dim for i, cur_dim in enumerate(array.shape) if i not in axis])
        extra_shape = tuple([cur_dim for i, cur_dim in enumerate(array.shape) if i in axis])

        if not isinstance(default, Tensor):
            default = full_array(target_shape, default)

        if default.shape != target_shape:
            raise ValueError("Wrong default arg in agg operation")

        index = [slice(None, None, None) for _ in range(array.n_dims)]
        for extra_index in Tensor.__static_full_search(extra_shape):
            for i, cur_dim in enumerate(axis):
                index[cur_dim] = extra_index[i]

            default = Tensor.__array_operation(default, array[tuple(index)], operation)

        return default

    def min(
            self, axis: Union[None, 'Tensor', Iterable, a_value_type] = None,
            default: Union[None, 'Tensor', a_value_type] = None
    ) -> Union['Tensor', value_type]:
        if default is None:
            default = Value(float('inf'))
        operation = lambda x, y: min(x, y)

        return Tensor.__aggregation_operation(self, operation, default, axis)

    def max(
            self, axis: Union[None, 'Tensor', Iterable, a_value_type] = None,
            default: Union[None, 'Tensor', a_value_type] = None
    ) -> Union['Tensor', value_type]:
        if default is None:
            default = Value(-float('inf'))
        operation = lambda x, y: max(x, y)

        return Tensor.__aggregation_operation(self, operation, default, axis)

    def sum(
            self, axis: Union[None, 'Tensor', a_value_type] = None,
            default: Union[None, 'Tensor', a_value_type] = None
    ) -> Union['Tensor', value_type]:
        if default is None:
            default = Value(0)
        operation = lambda x, y: x + y

        return Tensor.__aggregation_operation(self, operation, default, axis)

    def mean(
            self, axis: Union[None, 'Tensor', Iterable, a_value_type] = None,
    ) -> Union['Tensor', value_type]:
        if axis is None:
            axis = tuple(range(self.n_dims))
        elif isinstance(axis, int):
            axis = (axis,)
        divider = 1
        for cur_dim in axis:
            divider *= self.shape[cur_dim]

        return self.sum(axis=axis) / divider

    def prod(
            self, axis: Union[None, 'Tensor', a_value_type] = None,
    ) -> Union['Tensor', value_type]:
        operation = lambda x, y: x * y

        return Tensor.__aggregation_operation(self, operation, Value(1), axis)


def full_array(shape: tuple[int, ...], value: Union[a_value_type, Callable[[], a_value_type]]) -> Tensor:
    if len(shape) == 0:
        return Tensor([])

    if len(shape) == 1:
        return Tensor([value() if isinstance(value, Callable) else value for _ in range(shape[0])])

    return Tensor([full_array(shape[1:], value) for _ in range(shape[0])])


def zeros(shape: tuple[int, ...]) -> Tensor:
    return full_array(shape, 0)


def ones(shape: tuple[int, ...]) -> Tensor:
    return full_array(shape, 1)


def eye(shape: Union[tuple[int, ...], int]) -> Tensor:
    if isinstance(shape, int):
        shape = (shape, shape)
    eye_array = zeros(shape)
    for i in range(min(shape)):
        eye_array[tuple([i] * len(shape))] += 1

    return eye_array


def arange(start: a_value_type, stop: Optional[a_value_type] = None, step: Optional[a_value_type] = None) -> Tensor:
    if stop is None and step is None:
        stop = start
        start = 0
        step = 1 if stop > 0 else -1
    elif step is None:
        step = 1 if stop > start else -1
    else:
        raise ValueError('Wrong args in arange!')

    answer = [start]
    if start < stop:
        if step < 0:
            return Tensor([])
        answer = [start]
        while answer[-1] + step < stop:
            answer.append(answer[-1] + step)
    elif start > stop:
        if step > 0:
            return Tensor([])
        while answer[-1] + step > stop:
            answer.append(answer[-1] + step)
    else:
        return Tensor([])

    return Tensor(answer)


def concatenate(*arrays: Tensor, dim: int = 0) -> Tensor:
    shapes = []
    sum_current_dim_shape = 0
    for array in arrays:
        cur_shape = list(array.shape)
        if dim >= array.n_dims:
            raise ValueError("dim >= array.n_dims")

        sum_current_dim_shape += cur_shape[dim]
        cur_shape[dim] = 0
        shapes.append(tuple(cur_shape))

    shapes_set = set(shapes)
    if len(shapes_set) > 1:
        raise ValueError("Wrong shapes in concatenate!")

    target_shape = list(shapes_set.pop())
    target_shape[dim] = sum_current_dim_shape
    target_shape = tuple(target_shape)

    concatenated_array = zeros(target_shape)
    index = [slice(None, None, None) for _ in range(len(target_shape))]
    prev_sum = 0
    for array in arrays:
        index[dim] = slice(prev_sum, prev_sum + array.shape[dim])
        prev_sum += array.shape[dim]
        concatenated_array[tuple(index)] = array

    return concatenated_array
