from dataclasses import dataclass
from typing import Optional, Union, Iterable, get_args
from functools import lru_cache
from enum import Enum

import creampy
from creampy import Tensor, a_value_type, Value


def __is_vector(array: Tensor) -> bool:
    return len(array.shape) == 1 or (array.shape[0] == 1 and __is_matrix(array))


def __is_matrix(array: Tensor) -> bool:
    return isinstance(array, Tensor) and len(array.shape) == 2


def __is_square_matrix(array: Tensor) -> bool:
    return __is_matrix(array) and array.shape[0] == array.shape[1]


def matrix_exception(matrix: Tensor):
    if not __is_matrix(matrix):
        raise ValueError("Excepted matrix")


def square_matrix_exception(matrix: Tensor):
    matrix_exception(matrix)
    if not __is_square_matrix(matrix):
        raise ValueError("Excepted square matrix")


def not_singular_matrix_exception(matrix: Tensor):
    matrix_exception(matrix)
    square_matrix_exception(matrix)
    if rank(matrix) != min(matrix.shape):
        raise ValueError("Excepted not singular matrix")


def vector_exception(vector: Tensor):
    if not __is_vector(vector):
        raise ValueError("Excepted vector")


def dot(a: Tensor, b: Tensor) -> Union[Tensor, a_value_type]:
    return a @ b


def power(a: Union[Tensor, a_value_type], value: int) -> Union[Tensor, a_value_type]:
    if isinstance(a, get_args(a_value_type)):
        operation = lambda left, right: left * right
        one = Value(1)
    elif isinstance(a, Tensor):
        square_matrix_exception(a)
        operation = lambda left, right: dot(left, right)
        one = creampy.eye(a.shape)
    else:
        raise ValueError("Value in power must be Matrix or Number")

    if value == 0:
        return one

    if value % 2 == 0:
        q = power(a, value // 2)
        return operation(q, q)

    q = power(a, value - 1)
    return operation(q, a)


def vector_proj(vector: Tensor, axis: Tensor) -> Tensor:
    __is_vector(vector)
    __is_vector(axis)

    k = (dot(vector.reshape(1, -1), axis.reshape(-1, 1))
         / dot(axis.reshape(1, -1), axis.reshape(-1, 1)))

    return k * axis.copy()


def __orthogonalization(*vectors: Tensor, to_norm: bool = True) -> list[Tensor]:
    _vectors = [vector.copy().reshape(-1) for vector in vectors]
    orto_vectors = [normalize(_vectors[0]) if to_norm else _vectors[0]]
    for i in range(1, len(vectors)):
        target = _vectors[i]
        for j in range(i):
            target = target - vector_proj(target, orto_vectors[j])
        if to_norm:
            orto_vectors.append(normalize(target))
        else:
            orto_vectors.append(target)

    return orto_vectors


def orthogonalization(vectors: Union[Iterable[Tensor], Tensor], to_norm: bool = True) -> Tensor:
    if isinstance(vectors, Tensor):
        matrix_exception(vectors)
        return Tensor(__orthogonalization(*vectors.array, to_norm=to_norm))
    else:
        [vector_exception(vector) for vector in vectors]
        return Tensor(__orthogonalization(*vectors, to_norm=to_norm))


@dataclass
class QRDecompositionResult:
    q: Tensor
    r: Tensor

    def __iter__(self):
        return iter((self.q, self.r))


@dataclass
class HessenbergResult:
    matrix: Tensor
    transform_matrix: Tensor

    def __iter__(self):
        return iter((self.matrix, self.transform_matrix))


def __get_g_matrix(row1: int, row2: int, column: int, matrix: Tensor) -> Tensor:
    g_matrix = creampy.eye(matrix.shape)
    length = norm(matrix[[row1, row2], column].reshape(-1))
    g_matrix[row1, row1] = matrix[row1, column] / length
    g_matrix[row2, row2] = matrix[row1, column] / length
    g_matrix[row1, row2] = matrix[row2, column] / length
    g_matrix[row2, row1] = -matrix[row2, column] / length

    return g_matrix


def hessenberg(matrix: Tensor) -> HessenbergResult:
    __is_square_matrix(matrix)

    _matrix = matrix.copy()
    transform_matrix = creampy.eye(_matrix.shape)

    for column in range(matrix.shape[0] - 2):
        for row in range(column + 2, matrix.shape[0]):
            g_matrix = __get_g_matrix(column + 1, row, column, _matrix)
            _matrix = (g_matrix @ _matrix) @ g_matrix.T
            transform_matrix = g_matrix @ transform_matrix

    return HessenbergResult(_matrix, transform_matrix)


def qr(matrix: Tensor, use_hessenberg: bool = False) -> QRDecompositionResult:
    if not use_hessenberg:
        q = orthogonalization(matrix.T).T
        r = dot(q.T, matrix)
    else:
        q = creampy.eye(matrix.shape)
        r = matrix.copy()

        for i in range(r.shape[0] - 1):
            g_matrix = __get_g_matrix(i, i + 1, i, r)
            q = g_matrix @ q
            r = g_matrix @ r
        q = q.T

    q.clean()
    r.clean()

    return QRDecompositionResult(q, r)


class EigValsFindMethod(Enum):
    DEFAULT = 0
    HESSENBERG = 1
    BISECTION = 2


@dataclass
class EigResults:
    eig_values: Tensor
    eig_vectors: Tensor


def eigvals(matrix: Tensor, method: EigValsFindMethod = EigValsFindMethod.HESSENBERG) -> Tensor:
    square_matrix_exception(matrix)

    _matrix = matrix.copy()

    if method == EigValsFindMethod.DEFAULT:
        for i in range(_matrix.shape[0] ** 4):
            q, r = qr(_matrix)
            _matrix = r @ q

    elif method == EigValsFindMethod.HESSENBERG:
        _matrix, transform_matrix = hessenberg(_matrix)

        for _ in range(_matrix.shape[0] ** 3 * 2):
            q, r = qr(_matrix, use_hessenberg=True)
            _matrix = r @ q

    index = 0
    eigen_values = []
    while index < _matrix.shape[0]:
        if index == _matrix.shape[0] - 1 or _matrix[index + 1, index] == 0:
            eigen_values.append(_matrix[index, index])
            index += 1
        else:
            under_matrix = _matrix[index:index + 2, index:index + 2]
            B = -under_matrix.trace
            C = det(under_matrix)
            D = B ** 2 - 4 * C
            eigen_values += [Value((-B + D ** 0.5) / 2), Value((-B - D ** 0.5) / 2)]
            index += 2

    eigen_values = Tensor(eigen_values, check_complex=True)
    eigen_values.clean()

    return eigen_values


def eig(matrix: Tensor, method: EigValsFindMethod = EigValsFindMethod.HESSENBERG) -> EigResults:
    square_matrix_exception(matrix)

    eig_values = eigvals(matrix, method=method)
    eig_vectors = []

    for eig_value in eig_values.array:
        _matrix = matrix.copy()
        if isinstance(eig_value.value, complex):
            _matrix = _matrix.to_complex()
        _matrix = _matrix - eig_value * creampy.eye(matrix.shape)

        _solve = null_space(_matrix)
        eig_vectors.append(_solve)

    return EigResults(
        eig_values=eig_values,
        eig_vectors=Tensor(eig_vectors),
    )


@dataclass
class Operation:
    def apply_operation_to_matrix(self, matrix: Tensor, inplace: bool = False) -> Optional[Tensor]:
        pass


@dataclass
class PermuteOperation(Operation):
    first_row: int
    second_row: int

    def apply_operation_to_matrix(self, matrix: Tensor, inplace: bool = False) -> Optional[Tensor]:
        result = matrix.copy()
        result[[self.first_row, self.second_row]] = result[[self.second_row, self.first_row]].copy()
        if not inplace:
            return result
        matrix.update(result)


@dataclass
class MulOperation(Operation):
    target_row: int
    k: a_value_type

    def apply_operation_to_matrix(self, matrix: Tensor, inplace: bool = False) -> Optional[Tensor]:
        result = matrix.copy()
        result[self.target_row] *= creampy.Value(self.k)
        if not inplace:
            return result
        matrix.update(result)


@dataclass
class CombinedOperation(Operation):
    source_row: int
    target_row: int
    k: a_value_type

    def apply_operation_to_matrix(self, matrix: Tensor, inplace: bool = False) -> Optional[Tensor]:
        result = matrix.copy()
        result[self.target_row] += result[self.source_row] * creampy.Value(self.k)
        if not inplace:
            return result
        matrix.update(result)


def __apply_operations_to_matrix(matrix: Tensor, *operations: Operation, inplace: bool = False) -> Optional[Tensor]:
    matrix_exception(matrix)

    result = matrix.copy()
    for operation in operations:
        if not inplace:
            result = operation.apply_operation_to_matrix(result, False)
        else:
            operation.apply_operation_to_matrix(matrix, True)

    if not inplace:
        return result


@lru_cache
def stairs(
        matrix: Tensor, triangle: bool = False, save_operation: bool = False
) -> Union[Tensor, tuple[Tensor, tuple[Operation, ...]]]:
    operations_list = []
    result = matrix.copy()
    cur_row_index = 0
    for index in range(min(matrix.shape)):
        source_value = result[cur_row_index, index]
        if source_value == 0:
            for i in range(cur_row_index + 1, matrix.shape[0]):
                if result[i, index] != 0:
                    source_value = result[i, index]
                    operation = PermuteOperation(cur_row_index, i)
                    operation.apply_operation_to_matrix(result, True)
                    operations_list.append(operation)
                    break
        if source_value == 0:
            continue

        for i in range(cur_row_index + 1, matrix.shape[0]):
            if result[i, index] != 0:
                operation = CombinedOperation(cur_row_index, i, -result[i, index] / source_value)
                operation.apply_operation_to_matrix(result, True)
                operations_list.append(operation)
        cur_row_index += 1

    result.clean(100000 * creampy.EPS)
    if triangle:
        if save_operation:
            return result, tuple(operations_list)
        return result

    cur_col_index = matrix.shape[1]
    for row_index in range(cur_row_index - 1, -1, -1):
        for col_index in range(cur_col_index - 1, -1, -1):
            if col_index == 0 or result[row_index, col_index - 1] == 0:
                cur_col_index = col_index
                operation = MulOperation(row_index, 1 / result[row_index, col_index])
                operation.apply_operation_to_matrix(result, True)
                operations_list.append(operation)
                for i in range(row_index - 1, -1, -1):
                    operation = CombinedOperation(row_index, i, -result[i, col_index])
                    operation.apply_operation_to_matrix(result, True)
                    operations_list.append(operation)
                break

    result.clean(100000 * creampy.EPS)
    if save_operation:
        return result, tuple(operations_list)
    return result


def __is_zero_vector(vector: Tensor) -> bool:
    eps_k = 10000
    if not isinstance(vector[0].value, complex):
        return -creampy.EPS * eps_k <= vector.min() <= vector.max() <= creampy.EPS * eps_k
    vec_real = Tensor([value.value.real for value in vector.array])
    vec_imag = Tensor([value.value.imag for value in vector.array])
    return ((-creampy.EPS * eps_k <= vec_real.min() <= vec_real.max() <= creampy.EPS * eps_k)
            and (-creampy.EPS * eps_k <= vec_imag.min() <= vec_imag.max() <= creampy.EPS * eps_k))


def norm(vector: Tensor) -> a_value_type:
    vector_exception(vector)

    return (vector ** 2).sum() ** 0.5


def normalize(vector: Tensor) -> Tensor:
    vector_exception(vector)
    if __is_zero_vector(vector):
        return vector

    return vector / norm(vector)


@lru_cache
def rank(matrix: Tensor) -> int:
    matrix_exception(matrix)

    stairs_matrix = stairs(matrix, triangle=True)
    for index in range(matrix.shape[0], 0, -1):
        if not __is_zero_vector(stairs_matrix[index - 1]):
            return index

    return 0


@lru_cache
def det(matrix: Tensor) -> a_value_type:
    square_matrix_exception(matrix)

    stairs_matrix: Tensor = stairs(matrix, triangle=True)
    return stairs_matrix.main_diagonal().prod()


@lru_cache
def inv(matrix: Tensor) -> Tensor:
    not_singular_matrix_exception(matrix)
    _, operations = stairs(matrix, triangle=False, save_operation=True)
    eye = creampy.eye(matrix.shape)
    __apply_operations_to_matrix(eye, *operations, inplace=True)
    eye.clean()

    return eye


@lru_cache
def null_space(matrix: Tensor, linear_shell: bool = True, is_norm: bool = False) -> Tensor:
    matrix_exception(matrix)
    n = matrix.shape[1]
    if rank(matrix) == n:
        return creampy.zeros((n, 1))

    answer = creampy.zeros((n, n - rank(matrix)))
    stairs_matrix = stairs(matrix, triangle=True)
    stairs_matrix = stairs(stairs_matrix, triangle=False)
    for i in range(n - rank(matrix)):
        for j in range(rank(matrix)):
            answer[j, i] = -stairs_matrix[j, i + rank(matrix)]
        answer[rank(matrix) + i, i] = 1

    if not linear_shell:
        answer = answer.sum(axis=1).reshape(1, -1)

    if is_norm:
        for i in range(answer.shape[1]):
            answer[:, i] = normalize(answer[:, i])

    return answer


@dataclass
class SolveSOLE:
    kernel: Tensor
    shift: Tensor

    def __str__(self) -> str:
        return f'kernel:\n{str(self.kernel)}\nshift:\n{str(self.shift)}'


def solve(a: Tensor, b: Tensor) -> Optional[SolveSOLE]:
    matrix_exception(a)
    vector_exception(b)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    stairs_table, operations_1 = stairs(a, triangle=False, save_operation=True)
    stairs_table, operations_2 = stairs(stairs_table, triangle=True, save_operation=True)
    operations = tuple(list(operations_1) + list(operations_2))

    if rank(creampy.concatenate(a, b, dim=1)) > rank(a):
        return None

    kernel = null_space(a)
    stairs_b = __apply_operations_to_matrix(b, *operations, inplace=False)

    shift = creampy.zeros((b.shape[0], 1))
    for i in range(rank(creampy.concatenate(a, b, dim=1))):
        shift[i, 0] = stairs_b[i, 0]
    kernel.clean()
    shift.clean()

    return SolveSOLE(kernel, shift)