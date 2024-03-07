import pytest  # import but not used


def add(number1: int, number2: int) -> int:
    """number1とnumber2の加算

    Args:
        number1 (int): 整数
        number2 (int): 整数

    Returns:
        int: number1+number2
    """
    return number1 + number2


if __name__ == "__main__":
    print(add(2, 1))
