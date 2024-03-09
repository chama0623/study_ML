import numpy as np
from numpy import ndarray
from src.common.model import BaseModel


def check_is_train(func):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_train") or not self.is_train:
            raise ValueError(
                "The 'train' method must be executed before calling this method."
            )
        return func(self, *args, **kwargs)

    return wrapper


class MySimpleLinearRegression(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.coefficient = None  # 回帰係数(傾き)
        self.intercept = None  # y-切片

    def train(self, X: ndarray, y: ndarray) -> None:
        """最小二乗法により, 回帰係数とy-切片を求める


        Args:
            X (ndarray): 学習データ
            y (ndarray): 教師データ
        """
        # 分散共分散行列からVar(X), Cov(X, y)を求める
        cov_matrix = np.cov(X.T, y.T)
        var_x, cov_xy = tuple(map(tuple, cov_matrix))[0]
        # 回帰係数を求める
        self.coefficient = cov_xy / var_x

        # y-切片を求める
        self.intercept = np.mean(y) - self.coefficient * np.mean(X)
        self.is_train = True

    @check_is_train
    def predict(self, X: ndarray) -> ndarray:
        """単回帰モデルにより予測を行う

        Args:
            X (ndarray): 予測を行うデータ

        Returns:
            ndarray: 予測結果
        """
        return self.coefficient * X + self.intercept
