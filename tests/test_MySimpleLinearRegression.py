from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from src.regression.linear_regression_model import MySimpleLinearRegression


def test_MySimpleLinearRegression():
    error = 1e-15
    X, y = make_regression(n_samples=100, n_features=1, noise=1, random_state=0)
    print(X, y)
    # sklearn.linear_model.LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    # MySimpleLinearRegression
    my_model = MySimpleLinearRegression()
    my_model.train(X, y)

    assert lr_model.coef_[0] - my_model.coefficient < error
    assert lr_model.intercept_ - my_model.intercept < error
