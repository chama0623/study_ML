from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """機械学習モデルのベースとなるモデル"""

    def __init__(self) -> None:
        super().__init__()
        self.is_train = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """学習を行うメソッド

        Args:
            X (np.ndarray): 学習データ
            y (np.ndarray): 教師データ
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を行うメソッド

        Args:
            X (np.ndarray): 予測を行うデータ

        Returns:
            np.ndarray: 予測結果
        """
        pass

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """学習と予測をまとめて行うメソッド

        Args:
            X_train (np.ndarray): 学習データ
            y_train (np.ndarray): 教師データ
            X_test (np.ndarray): 予測を行うデータ

        Returns:
            np.ndarray: X_testに対する予測結果
        """
        self.train(X_train, y_train)
        return self.predict(X_test)
