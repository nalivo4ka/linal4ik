from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import creampy as cp
import creampy.random
import creampy.poly
import creampy.linalg


def gauss_solver(a: cp.Tensor, b: cp.Tensor) -> cp.linalg.SolveSOLE:
    return cp.linalg.solve(a, b)


def explained_variance_ratio(eig_values: cp.Tensor, k: int) -> float:
    return eig_values[:k].sum() / eig_values.sum()


def find_best_k(eig_values: cp.Tensor, threshold: float) -> int:
    return list(filter(lambda k: explained_variance_ratio(eig_values, k) > threshold, range(2, eig_values.shape[0] + 1)))[0]


@dataclass
class PCA:
    matrix: cp.Tensor

    def plot_pca_projection(self, index_proj_1: int, index_proj_2: int) -> Figure:
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(self.matrix[:, index_proj_1].to_list(), self.matrix[:, index_proj_2].to_list())
        ax.set_title('График двух проекций PCA датасета')
        ax.set_xlabel(f'Компонента номер {index_proj_1 + 1}')
        ax.set_ylabel(f'Компонента номер {index_proj_2 + 1}')

        return fig


@dataclass
class Dataset:
    columns: list[str]
    matrix: cp.Tensor

    @staticmethod
    def from_csv(filepath: str, sep: str = ';') -> 'Dataset':
        with open(filepath) as file:
            columns = file.readline().split(sep)
            dataset_dict = {_column: [] for _column in columns}
            _is_numeric = {_column: True for _column in columns}
            for row in file.readlines():
                values = row.split(sep)
                for _column, _value in zip(columns, values):
                    try:
                        _value = float(_value)
                    except Exception:
                        pass

                    if not isinstance(_value, float):
                        _is_numeric[_column] = False

                    dataset_dict[_column].append(_value)

        columns = list(filter(lambda _column: _is_numeric[_column], columns))[:7]
        matrix = cp.Tensor([dataset_dict[_column] for _column in columns]).T

        return Dataset(
            columns=columns,
            matrix=matrix,
        )

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def copy(self) -> 'Dataset':
        return Dataset(
            columns=self.columns.copy(),
            matrix=self.matrix.copy()
        )

    def fillna(self) -> 'Dataset':
        new_dataset = self.copy()
        for col_index in range(self.matrix.shape[1]):
            cur_column = new_dataset.matrix[:, col_index]
            nan_mask = cur_column.isna()
            if nan_mask.sum() > 0:
                new_dataset.matrix[:, col_index][nan_mask] = cur_column[~nan_mask].mean()

        return new_dataset

    def center_data(self) -> 'Dataset':
        new_dataset = self.copy()
        for col_index in range(self.matrix.shape[1]):
            new_dataset.matrix[:, col_index] -= new_dataset.matrix[:, col_index].mean()

        return new_dataset

    def covariance_matrix(self) -> cp.Tensor:
        return (self.matrix.T @ self.matrix) / (len(self) - 1)

    def pca(self, k: Optional[int] = None, threshold: Optional[float] = None) -> PCA:
        new_dataset = self.copy()
        new_dataset = new_dataset.center_data()
        new_dataset = new_dataset.fillna()
        cov = new_dataset.covariance_matrix()
        eig_results = cp.linalg.eig(matrix=cov)
        if threshold is not None:
            k = find_best_k(eig_values=eig_results.eig_values, threshold=threshold)
            v = eig_results.eig_vectors[:k, :, 0].T
        elif k is not None:
            v = eig_results.eig_vectors[:k, :, 0].T
        else:
            raise ValueError

        for i in range(v.shape[0]):
            v[i] = cp.linalg.normalize(v[i])
        v = v.T

        return PCA(new_dataset.matrix @ v)

    def add_noise_and_compare(self, noise_level: float = 0.1) -> tuple[PCA, PCA]:
        new_dataset = self.copy()
        std_init = ((new_dataset.matrix - new_dataset.matrix.mean()) ** 2).sum()
        noise_dataset = Dataset(
            columns=new_dataset.columns,
            matrix=new_dataset.matrix + cp.random._random(new_dataset.matrix.shape) * std_init * noise_level
        )

        default_pca = new_dataset.pca(threshold=0.5)
        noice_pca = noise_dataset.pca(threshold=0.5)

        return default_pca, noice_pca


@dataclass
class RandomDataset(Dataset):
    n: int
    m: int
    columns: list[str] = field(init=False)
    matrix: cp.Tensor = field(init=False)

    def __post_init__(self):
        self.columns = [f'column_{i + 1}' for i in range(self.n)]
        self.matrix = cp.random._random((self.n, self.m))


def main():
    random_dataset = RandomDataset(1000, 6)
    normal_dataset = Dataset.from_csv('/Users/karpov0o0/Downloads/housing.csv', sep=',')
    pca = normal_dataset.pca(threshold=0.95)
    fig = pca.plot_pca_projection(0, 1)
    plt.show()


if __name__ == '__main__':
    main()
